import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import models
from timm import create_model

class ViT_B_WithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain): 
        super(ViT_B_WithIntermediate, self).__init__()
        self.stage = stage
        if pretrain=='22k':
            model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
            self.vit = create_model(model_name, pretrained=True).to(device)
            print(f"Using {model_name} with DINOv2 pretrained weights")
        elif pretrain=='1k':
            model_name = "vit_base_patch16_384.augreg_in1k"
            self.vit = create_model(model_name, pretrained=True).to(device)
            print(f"Using {model_name} with ImageNet-1k pretrained weights")
        self.embed_dim = self.vit.embed_dim   
        self.scale  = self.vit.patch_embed.grid_size[0]
        self.cat_token_num = 0
        if not self.vit.cls_token is None:
            self.cat_token_num += self.vit.cls_token.shape[1]
        if not self.vit.reg_token is None:
            self.cat_token_num += self.vit.reg_token.shape[1]

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        intermediate_results = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in {3, 6, 9}:
                t = x.clone()[:,self.cat_token_num:,:]
                t = t.permute(0, 2, 1).reshape(batch_size, self.embed_dim, self.scale, self.scale)
                intermediate_results.append(t)

        x = x[:, self.cat_token_num:, :]
        x = x.permute(0, 2, 1).reshape(batch_size, self.embed_dim, self.scale, self.scale)
        if self.stage==1:
            return x
        else:
            return x, intermediate_results
        
class Swin_B_WithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain): 
        super(Swin_B_WithIntermediate, self).__init__()
        self.stage = stage
        if pretrain=='22k':
            model_name = "swin_base_patch4_window12_384.ms_in22k"
        elif pretrain=='1k':
            model_name = "swin_base_patch4_window12_384.ms_in1k"
        self.swin = create_model(model_name, pretrained=True).to(device)
        self.embed_dim = self.swin.num_features  
        self.scale  = self.swin.patch_embed.grid_size[0]
       
    def forward(self, x):
        x = self.swin.patch_embed(x)
        intermediate_results = []
        batch_size, _, _, _ = x.shape
        for i, block in enumerate(self.swin.layers):
            x = block(x)
            if i in {0, 1, 2}:
                t = x.clone().permute(0, 3, 1, 2)
                intermediate_results.append(t)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2)

        if self.stage==1:
            return x
        else:
            return x, intermediate_results

class tresnet_WithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain): 
        super(tresnet_WithIntermediate, self).__init__()
        self.stage = stage
        if pretrain=='22k':
            model_name = "tresnet_v2_l.miil_in21k"
        elif pretrain=='1k':
            model_name = "tresnet_l.miil_in1k"
        self.tresnet = create_model(model_name, pretrained=True).to(device)

    def forward(self, x):
        intermediate_results = []
        x = self.tresnet.body.s2d(x)
        x = self.tresnet.body.conv1(x)
        x = self.tresnet.body.layer1(x)
        intermediate_results.append(x.clone())
        x = self.tresnet.body.layer2(x)
        intermediate_results.append(x.clone())
        x = self.tresnet.body.layer3(x)
        intermediate_results.append(x.clone())
        x = self.tresnet.body.layer4(x)
        if self.stage==1:
            return x
        else:
            return x, intermediate_results

class ResNet50WithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain='1k'): 
        super(ResNet50WithIntermediate, self).__init__()
        self.stage = stage
        self.resnet50 = create_model("resnet50.tv2_in1k", pretrained=True).to(device)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.act1(x)
        x = self.resnet50.maxpool(x)

        intermediate_results = []
        x = self.resnet50.layer1(x)
        intermediate_results.append(x.clone())
        x = self.resnet50.layer2(x)
        intermediate_results.append(x.clone())
        x = self.resnet50.layer3(x)
        intermediate_results.append(x.clone())
        x = self.resnet50.layer4(x)

        if self.stage==1:
            return x
        else:
            return x, intermediate_results
        

class XceptionWithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain='1k'): 
        super(XceptionWithIntermediate, self).__init__()
        self.stage = stage
        self.Xception = create_model("xception", pretrained=True).to(device)

    def forward(self, x):
        x = self.Xception.conv1(x)
        x = self.Xception.bn1(x)
        x = self.Xception.act1(x)

        x = self.Xception.conv2(x)
        x = self.Xception.bn2(x)
        x = self.Xception.act2(x)

        intermediate_results = []

        x = self.Xception.block1(x)
        x = self.Xception.block2(x)
        x = self.Xception.block3(x)
        intermediate_results.append(x.clone())

        x = self.Xception.block4(x)
        x = self.Xception.block5(x)
        x = self.Xception.block6(x)
        intermediate_results.append(x.clone())

        x = self.Xception.block7(x)
        x = self.Xception.block8(x)
        x = self.Xception.block9(x)
        intermediate_results.append(x.clone())

        x = self.Xception.block10(x)
        x = self.Xception.block11(x)
        x = self.Xception.block12(x)

        x = self.Xception.conv3(x)
        x = self.Xception.bn3(x)
        x = self.Xception.act3(x)

        x = self.Xception.conv4(x)
        x = self.Xception.bn4(x)
        x = self.Xception.act4(x)

        if self.stage == 1:
            return x
        else:
            return x, intermediate_results


class EfficientNetB7WithIntermediate(nn.Module):
    def __init__(self, device, stage, pretrain='1k'):
        super(EfficientNetB7WithIntermediate, self).__init__()
        self.stage = stage
        # Use timm's TF EfficientNet-B7 (Noisy Student) variant
        self.effnet = create_model("tf_efficientnet_b7_ns", pretrained=True).to(device)

    def forward(self, x):
        # EfficientNet in timm exposes forward_features for the final feature map
        feats = self.effnet.forward_features(x)
        if self.stage == 1:
            return feats
        else:
            # Minimal compatibility: return empty intermediates for multi-stage
            intermediate_results = []
            return feats, intermediate_results