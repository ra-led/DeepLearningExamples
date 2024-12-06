# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None, weights="IMAGENET1K_V1"):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        # x = self.feature_extractor(x)
        # return x
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            features.append(x)
        return features


class SSD300(nn.Module):
    def __init__(self, backbone=ResNet('resnet50')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).reshape(s.size(0), 4, -1), c(s).reshape(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*((mask + neg_mask).float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret


class FeatureFusionSSD300(SSD300):
    def __init__(self, backbone=ResNet('resnet50'), fusion_type='concat'):
        super().__init__(backbone)
        self.fusion_type = fusion_type
        self.out_channels = backbone.out_channels
        self.fusion_layers = nn.ModuleList()

        conv4_3_channels = 256
        conv5_3_channels = 512

        if fusion_type == 'concat':
            # Upsample conv5_3 to match conv4_3 size
            self.fusion_layers_5_3 = nn.Sequential(
                nn.ConvTranspose2d(conv5_3_channels, conv4_3_channels, kernel_size=2, stride=2),
                nn.Conv2d(conv4_3_channels, conv4_3_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv4_3_channels),
                nn.ReLU(inplace=True)
            )
            # conv4_3 learning the better features to fuse
            self.fusion_layers_4_3 = nn.Sequential(
                nn.Conv2d(conv4_3_channels, conv4_3_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv4_3_channels),
                nn.ReLU(inplace=True)
            )
            # Concatenate and reduce channels
            self.fusion_layers_con = nn.Conv2d(conv4_3_channels*2, conv4_3_channels, kernel_size=1)
        elif fusion_type == 'element-sum':
            # Upsample conv5_3 to match conv4_3 size
            self.fusion_layers_5_3 = nn.Sequential(
                nn.ConvTranspose2d(conv5_3_channels, conv4_3_channels, kernel_size=2, stride=2),
                nn.Conv2d(conv4_3_channels, conv4_3_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv4_3_channels),
            )
            # conv4_3 learning the better features to fuse
            self.fusion_layers_4_3 = nn.Sequential(
                nn.Conv2d(conv4_3_channels, conv4_3_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(conv4_3_channels),
            )
            self.fusion_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        features = self.feature_extractor(x)
        # Assume features is a list of feature maps from the backbone
        # features[2] is conv4_3 equivalent, features[3] is conv5_3 equivalent
        if self.fusion_type == 'concat':
            # Fuse layer2 and layer3
            fused = torch.cat(
                (
                    self.fusion_layers_4_3(features[-3]),
                    self.fusion_layers_5_3(features[-2])
                ),
                dim=1
            )
            fused = self.fusion_layers_con(fused)
        elif self.fusion_type == 'element-sum':
            # Fuse layer2 and layer3
            fused = self.fusion_layers_4_3(features[-2]) + self.fusion_layers_5_3(features[-1])
            fused = self.fusion_activation(fused)

        additional_x = features[-1]    
        detection_feed = [fused, additional_x]  # Start from the fused layer
        for l in self.additional_blocks:
            additional_x = l(additional_x)
            detection_feed.append(additional_x)

        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
