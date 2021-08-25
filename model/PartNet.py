import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
import math

import model.resnet as models
import model.vgg as vgg_models


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class PFENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(PFENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query_level_2 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp_level_2 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )



        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(50 + 1, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.part_num = 50

        self.alpha = nn.Linear(256, 256)
        self.beta = nn.Linear(256, 256)

        variance = math.sqrt(1.0)

        self.part_emb_2_level = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))

        # self.key = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
     


    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_level_2 = self.down_query_level_2(query_feat)

        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        supp_feat_level_2_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_level_2 = self.down_supp_level_2(supp_feat)
            supp_feat_level_2_list.append(supp_feat_level_2)
            # spatial_supp_feat = supp_feat
            # supp_feat = Weighted_GAP(supp_feat, mask)
            # supp_feat_list.append(supp_feat)


        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat_level_2.size(2), query_feat_level_2.size(3)), mode='bilinear', align_corners=True)

        # if self.shot > 1:
        #     supp_feat = supp_feat_list[0]
        #     for i in range(1, len(supp_feat_list)):
        #         supp_feat += supp_feat_list[i]
        #     supp_feat /= len(supp_feat_list)

        out_list = []
        pyramid_feat_list = []
        re_sim_list = []
        query_supp_sim_weight_list = []

        ################################################################
        #                part emb sim                                  #
        ################################################################

        if self.shot > 1:


            part_embeds_2_level = self.part_emb_2_level.repeat(bsize, 1, 1)

            # cls_emb_key = self.key(part_embeds.unsqueeze(-1)).squeeze(-1).transpose(1, 2) # bs * 20 * 256

            part_emb_key_level_2 = part_embeds_2_level.transpose(1, 2)  # bs * 20 * 256

            for i, supp_feat_level_2 in enumerate(supp_feat_level_2_list):

                spatial_supp_feat_level_2 = supp_feat_level_2.view(bsize, supp_feat_level_2.size(1), -1)  # bs * 256 * 3600

                part_supp_sim_level_2 = torch.bmm(part_emb_key_level_2, spatial_supp_feat_level_2)  # bs * 20 * 3600

                if i == 0:
                    recon_supp_feat_level_2 = torch.bmm(part_emb_key_level_2.transpose(1, 2),
                                                        F.softmax(part_supp_sim_level_2, dim=1))

                    recon_ori_sim_level_2 = torch.bmm(recon_supp_feat_level_2.transpose(1, 2),
                                                      spatial_supp_feat_level_2)

                    sim_gt = torch.linspace(0, 3599, steps=3600).unsqueeze(0).repeat(bsize, 1).cuda()

                    sim_loss_level_2 = F.cross_entropy(recon_ori_sim_level_2, sim_gt.long()) * 0.1

                part_supp_sim_level_2 = part_supp_sim_level_2.view(bsize, part_supp_sim_level_2.size(1), 60,
                                                                   60)  # bs * 20 * 60 * 60


                part_supp_sim_level_2 = part_supp_sim_level_2.view(bsize, self.part_num, -1)

                part_supp_sim_level_2_norm = torch.norm(part_supp_sim_level_2, 2, 1, True)


                part_query_sim_level_2 = torch.bmm(query_feat_level_2.view(bsize, 256, -1).transpose(1, 2),
                                                   part_emb_key_level_2.transpose(1, 2))  # bs * 3600 * 20


                part_query_sim_level_2_norm = torch.norm(part_query_sim_level_2, 2, 2, True)

                query_supp_similarity = torch.bmm(part_query_sim_level_2, part_supp_sim_level_2, ) \
                                        / (torch.bmm(part_query_sim_level_2_norm, part_supp_sim_level_2_norm) + cosine_eps)

                query_supp_similarity = query_supp_similarity * mask.view(bsize, 1, -1)

                query_supp_similarity = query_supp_similarity.masked_fill(query_supp_similarity == 0, -1e9)

                query_supp_similarity_soft = F.softmax(query_supp_similarity, dim=-1)

                query_supp_similarity_sig = F.sigmoid(query_supp_similarity)

                query_supp_similarity_weight = query_supp_similarity_sig.sum(-1)

                part_supp_sim_level_2_sig = F.sigmoid(part_supp_sim_level_2)

                re_sim = torch.bmm(query_supp_similarity_soft, part_supp_sim_level_2_sig.transpose(1, 2))

                part_query_sim_level_2 = nn.functional.sigmoid(part_query_sim_level_2)

                # weighted_SHOW = (SHOW * fg_weight.unsqueeze(1).unsqueeze(1)).permute(0, 1, 3, 2)
                part_query_sim_map_level_2 = (part_query_sim_level_2).transpose(1, 2)
                re_sim = re_sim.transpose(1, 2)
                re_sim = re_sim.view(bsize, self.part_num, 60, 60)

                re_sim_list.append(re_sim.unsqueeze(1))
                query_supp_sim_weight_list.append(query_supp_similarity_weight.unsqueeze(1))

            query_supp_sim_weight_cat = torch.cat(query_supp_sim_weight_list, 1)

            query_supp_sim_weight = F.softmax(query_supp_sim_weight_cat, dim=1).view(bsize, self.shot, 60, 60)

            re_sim_weighted_mean = (torch.cat(re_sim_list, 1) * query_supp_sim_weight.unsqueeze(2)).sum(1)

            part_query_sim_map_level_2 = part_query_sim_map_level_2.view(bsize, self.part_num, 60, 60)

            ori_part_query_sim_map_level_2 = part_query_sim_map_level_2

            part_query_sim_map_level_2 = part_query_sim_map_level_2 * re_sim_weighted_mean
        else:

            part_embeds_2_level = self.part_emb_2_level.repeat(bsize, 1, 1)

            # cls_emb_key = self.key(part_embeds.unsqueeze(-1)).squeeze(-1).transpose(1, 2) # bs * 20 * 256

            part_emb_key_level_2 = part_embeds_2_level.transpose(1, 2)  # bs * 20 * 256

            for i, supp_feat_level_2 in enumerate(supp_feat_level_2_list):
                spatial_supp_feat_level_2 = supp_feat_level_2.view(bsize, supp_feat_level_2.size(1),
                                                                   -1)  # bs * 256 * 3600

                part_supp_sim_level_2 = torch.bmm(part_emb_key_level_2, spatial_supp_feat_level_2)  # bs * 20 * 3600

                recon_supp_feat_level_2 = torch.bmm(part_emb_key_level_2.transpose(1, 2),
                                                    F.softmax(part_supp_sim_level_2, dim=1))

                recon_ori_sim_level_2 = torch.bmm(recon_supp_feat_level_2.transpose(1, 2), spatial_supp_feat_level_2)

                sim_gt = torch.linspace(0, 3599, steps=3600).unsqueeze(0).repeat(bsize, 1).cuda()

                supp_sim_loss_level_2 = F.cross_entropy(recon_ori_sim_level_2, sim_gt.long()) * 0.1



                part_supp_sim_level_2 = part_supp_sim_level_2.view(bsize, part_supp_sim_level_2.size(1), 60,
                                                                   60)  # bs * 20 * 60 * 60

                part_supp_sim_level_2 = part_supp_sim_level_2.view(bsize, self.part_num, -1)

                part_supp_sim_level_2_norm = torch.norm(part_supp_sim_level_2, 2, 1, True)

                part_query_sim_level_2 = torch.bmm(query_feat_level_2.view(bsize, 256, -1).transpose(1, 2),
                                                   part_emb_key_level_2.transpose(1, 2))  # bs * 3600 * 20

                part_query_sim_level_2_norm = torch.norm(part_query_sim_level_2, 2, 2, True)

                query_supp_similarity = torch.bmm(part_query_sim_level_2, part_supp_sim_level_2, ) \
                                        / (torch.bmm(part_query_sim_level_2_norm,
                                                     part_supp_sim_level_2_norm) + cosine_eps)

                query_supp_similarity = query_supp_similarity * mask.view(bsize, 1, -1)

                query_supp_similarity = query_supp_similarity.masked_fill(query_supp_similarity == 0, -1e9)

                query_supp_similarity_soft = F.softmax(query_supp_similarity, dim=-1)

                query_supp_similarity_sig = F.sigmoid(query_supp_similarity)

                query_supp_similarity_weight = query_supp_similarity_sig.sum(-1)

                part_supp_sim_level_2_sig = F.sigmoid(part_supp_sim_level_2)

                re_sim = torch.bmm(query_supp_similarity_soft, part_supp_sim_level_2_sig.transpose(1, 2))

                part_query_sim_level_2 = nn.functional.sigmoid(part_query_sim_level_2)

                # weighted_SHOW = (SHOW * fg_weight.unsqueeze(1).unsqueeze(1)).permute(0, 1, 3, 2)
                part_query_sim_map_level_2 = (part_query_sim_level_2).transpose(1, 2)
                re_sim = re_sim.transpose(1, 2)
                re_sim = re_sim.view(bsize, self.part_num, 60, 60)

                re_sim_list.append(re_sim.unsqueeze(1))
                query_supp_sim_weight_list.append(query_supp_similarity_weight.unsqueeze(1))

            query_supp_sim_weight_cat = torch.cat(query_supp_sim_weight_list, 1)

            query_supp_sim_weight = F.softmax(query_supp_sim_weight_cat, dim=1).view(bsize, self.shot, 60, 60)

            re_sim_weighted_mean = (torch.cat(re_sim_list, 1) * query_supp_sim_weight.unsqueeze(2)).sum(1)

            part_query_sim_map_level_2 = part_query_sim_map_level_2.view(bsize, self.part_num, 60, 60)

            ori_part_query_sim_map_level_2 = part_query_sim_map_level_2

            part_query_sim_map_level_2 = part_query_sim_map_level_2 * re_sim_weighted_mean

            # caculate loss

            recon_query_feat_level_2 = torch.bmm(part_emb_key_level_2.transpose(1, 2),
                                                 F.softmax(part_query_sim_level_2.transpose(1, 2), dim=1))

            recon_ori_sim_level_2 = torch.bmm(recon_query_feat_level_2.transpose(1, 2),
                                              query_feat_level_2.view(bsize, 256, -1))

            sim_gt = torch.linspace(0, 3599, steps=3600).unsqueeze(0).repeat(bsize, 1).cuda()

            query_sim_loss_level_2 = F.cross_entropy(recon_ori_sim_level_2, sim_gt.long()) * 0.1


        ###############################################################################
        #                          start decode                                       #
        ###############################################################################



        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat_level_2.shape[2] * tmp_bin)
            else:
                bin = tmp_bin
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            part_mask_level_2_bin = F.interpolate(part_query_sim_map_level_2, size=(bin, bin), mode='bilinear', align_corners=True)

            merge_feat_bin = torch.cat([part_mask_level_2_bin, corr_mask_bin], 1)
            # merge_feat_bin = part_mask_bin
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat_level_2.size(2), query_feat_level_2.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)
        

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            weighted_part_map_level_2 = F.interpolate(part_query_sim_map_level_2, size=(h, w), mode='bilinear', align_corners=True)
            ori_part_map_level_2 = F.interpolate(ori_part_query_sim_map_level_2, size=(h, w), mode='bilinear', align_corners=True)


        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()    

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss, sim_loss_level_2
        else:
            return out





