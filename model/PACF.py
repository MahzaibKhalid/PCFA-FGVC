import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
from .loss import CrossEntropyLabelSmooth
from .model import ResNet50WithIntermediate, ViT_B_WithIntermediate, Swin_B_WithIntermediate, tresnet_WithIntermediate, XceptionWithIntermediate, EfficientNetB7WithIntermediate

class PACF_model(nn.Module):
    def __init__(
        self,
        device,
        part_name,
        stage=4,
        backbone='resnet50',
        dataset_name='Aircraft',
        dropout_p=0.1,
        embed_dim=512,
        to_scale=14,
        high_resolution_size=448,
        seg_atten_size=224,
        depth=1,
        use_precomputed_masks=False,
        part_mask_path=None,
    ):
        super().__init__()
        self.device = device
        self.stage = stage
        self.use_precomputed_masks = use_precomputed_masks
        
        if use_precomputed_masks:
            # Load pre-computed part masks
            self.part_masks = torch.load(part_mask_path, map_location=device)  # (N, 16, 16)
            self.num_parts = len(part_name)
        else:
            
            pass

        self.high_resolution_size = high_resolution_size
        self.dropout_p = dropout_p
        self.seg_atten_size = seg_atten_size
        self.embed_dim = embed_dim
        self.to_scale = to_scale

        # Load class information (robust to comma- or whitespace-separated formats)
        with open(f"datasets/SupplementaryData/{dataset_name}/class_type.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
            class_type = {}
            class_set = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                label_a = None
                label_b = None
                key = None
                # Prefer comma-separated if present (e.g., "0,dog breed,N02085620-Chihuahua")
                if ',' in line:
                    fields = [t.strip() for t in line.split(',')]
                    # Expected at least: index, mid-level label, fine/class or synset
                    if len(fields) >= 3:
                        key = fields[0]
                        label_a = fields[1]
                        label_b = fields[2]
                    elif len(fields) == 2:
                        key = fields[0]
                        label_a = fields[1]
                else:
                    tokens = line.split()
                    # Expected: index label_a label_b ... (take first three tokens if present)
                    if len(tokens) >= 3:
                        key = tokens[0]
                        label_a = tokens[1]
                        label_b = tokens[2]
                    elif len(tokens) == 2:
                        key = tokens[0]
                        label_a = tokens[1]

                # Fallbacks to avoid IndexErrors; skip malformed lines
                if key is None or label_a is None:
                    continue
                if label_b is None:
                    # Duplicate label_a if only one label available to keep downstream shape consistent
                    label_b = label_a

                class_set.append(label_a)
                class_set.append(label_b)
                class_type[key] = [label_a, label_b]
        self.class_set = sorted(list(set(class_set)))

        class_num = len(list(class_type.keys()))
        self.class_num = class_num
        self.class_value = list(class_type.values())

        epsilon = 0.7

        similarity_matrix = np.zeros((1, 2, class_num, class_num))
        for i in range(class_num):
            for j in range(class_num):
                if i == j:
                    similarity_matrix[0][1][i][j] = 1
                    similarity_matrix[0][0][i][j] = epsilon
                else:
                    similarity_matrix[0][0][i][j] = (1 - epsilon) / class_num

        num_heads = 8
        # Determine number of part branches
        if use_precomputed_masks:
            self.part_num = self.num_parts + 1  # +1 coarse
        else:
            # Default to 10 parts if no precomputed masks
            self.part_num = len(part_name) + 1        
        self.cls_token = nn.Parameter(torch.rand(self.part_num, 1, 3, 1, embed_dim).to(device))
        self.mid_class = {}
        for i, mid_class in enumerate(self.class_set):
            self.mid_class[mid_class] = i
        if dataset_name=='Dogs':
            pretrain='1k'
        else:
            pretrain='22k'
        if backbone=='resnet50':
            self.backbone = ResNet50WithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            input_dim = self.backbone.resnet50.num_features
            self.high_resolution_size = 448
            input_dim_1 = self.backbone.resnet50.feature_info[1]['num_chs']
            input_dim_2 = self.backbone.resnet50.feature_info[2]['num_chs']
            input_dim_3 = self.backbone.resnet50.feature_info[3]['num_chs']
            input_dim_4 = input_dim
        elif backbone=='vit-b':
            self.backbone = ViT_B_WithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            input_dim = self.backbone.vit.embed_dim   
            self.high_resolution_size = self.backbone.vit.pretrained_cfg['input_size'][1]
            input_dim_1 = input_dim_2 = input_dim_3 = input_dim_4 = input_dim
        elif backbone=='swin-b':
            self.backbone = Swin_B_WithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            input_dim = self.backbone.swin.num_features     
            self.high_resolution_size = self.backbone.swin.pretrained_cfg['input_size'][1]
            input_dim_1 = self.backbone.swin.feature_info[0]['num_chs']
            input_dim_2 = self.backbone.swin.feature_info[1]['num_chs']
            input_dim_3 = self.backbone.swin.feature_info[2]['num_chs']
            input_dim_4 = input_dim
        elif backbone=='tresnet':
            self.backbone = tresnet_WithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            input_dim = self.backbone.tresnet.num_features     
            self.high_resolution_size = 448
            input_dim_1 = self.backbone.tresnet.feature_info[1]['num_chs']
            input_dim_2 = self.backbone.tresnet.feature_info[2]['num_chs']
            input_dim_3 = self.backbone.tresnet.feature_info[3]['num_chs']
            input_dim_4 = input_dim
        elif backbone=='Xception':
            self.backbone = XceptionWithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            input_dim = self.backbone.Xception.num_features     
            self.high_resolution_size = 448
            input_dim_1 = input_dim_2 = input_dim_3 = self.backbone.Xception.feature_info[3]['num_chs']
            input_dim_4 = input_dim
        elif backbone=='efficientnet-b7':
            self.backbone = EfficientNetB7WithIntermediate(device, stage=stage, pretrain=pretrain).to(device)
            # timm efficientnet exposes num_features and pretrained_cfg for input size
            input_dim = self.backbone.effnet.num_features
            # Force EfficientNet-B7 input to 448x448 as requested
            self.high_resolution_size = 448
            # For stage==1 only input_dim is used; fill others safely
            input_dim_1 = input_dim_2 = input_dim_3 = input_dim_4 = input_dim
        else:
            warnings.warn("Undefined backbone", ImportWarning)
            return
        
        from .block import Conv
        self.MSCFS = nn.Sequential(
        nn.AdaptiveAvgPool2d((to_scale, to_scale)),
        Conv(input_dim, embed_dim, k=1, p=0)
        ).to(device)
         
        
        self.norm = nn.ModuleList(
            nn.LayerNorm(embed_dim)
            for i in range(self.part_num)).to(device)
        
        self.transformer = nn.ModuleList(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_p), 
                num_layers=depth
            )
            for i in range(self.part_num)).to(device)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_weights = [0.1, 0.2, 0.4, 1.0] 
        self.labelsmooth_criterion = CrossEntropyLabelSmooth(epsilon=torch.tensor(similarity_matrix[0]).to(device))
        self.cels = [self.labelsmooth_criterion]
        
        # Visual-only classifier for ablation (3 CLS tokens concatenated)
        self.classifier = nn.Linear(embed_dim * 3, self.class_num).to(device)
    
    def cluster_to_masks(self, cluster_labels, num_parts):
        """Convert cluster labels to binary masks with zero padding for missing clusters"""
        masks = []
        for part_id in range(num_parts):
            if part_id in cluster_labels:
                mask = (cluster_labels == part_id).float()
            else:
                mask = torch.zeros_like(cluster_labels, dtype=torch.float)
            masks.append(mask)
        return torch.stack(masks, dim=0)  # (num_parts, H, W)

    def forward(
        self,
        batch: dict   
    ):
        images = batch['img']
        img_tensor = torch.stack(images)
        bs = img_tensor.shape[0]
        
        if self.use_precomputed_masks:
            # Use pre-computed masks
            image_indices = batch['image_index'].to(self.device)
            cluster_labels = self.part_masks[image_indices]  # (B, 16, 16)
            
            # Convert cluster labels to binary masks
            atten_map = []
            for i in range(bs):
                masks = self.cluster_to_masks(cluster_labels[i], self.num_parts)  # (num_parts, 16, 16)
                # Resize to match expected size (16x16 -> 14x14)
                masks = nn.functional.interpolate(masks.unsqueeze(0), (self.to_scale, self.to_scale), mode='bilinear').squeeze(0)
                atten_map.append(masks)
            atten_map = torch.stack(atten_map)  # (B, num_parts, 14, 14)
            
        else:
            # Default to all ones mask if no precomputed masks
            # This is a placeholder - you may need to implement another mask generation method
            atten_map = torch.ones(bs, len(batch.get('part_name', [])), self.to_scale, self.to_scale).to(self.device)
        
        label_index = batch['label_index'].to(self.device)
        mask = (atten_map.sum(dim=(2, 3)) != 0).float()
        img_tensor = nn.functional.interpolate(img_tensor, (self.high_resolution_size, self.high_resolution_size), mode='bilinear')
        
        high_features = self.backbone(img_tensor)
        high_features = self.MSCFS(high_features)
        high_features = high_features.reshape(bs, high_features.shape[1], -1)
        high_features = high_features.permute(0, 2, 1)
        
        atten_map = atten_map.reshape(bs, atten_map.shape[1], -1)
        atten_map = atten_map.permute(0, 2, 1).unsqueeze(2)
        
        high_features = high_features.unsqueeze(-1)
        coarse_features = high_features.permute(3, 1, 0, 2)
        
        # THE KEY STEP: Hadamard product
        high_features = atten_map * high_features
        
        high_features = high_features.permute(3, 1, 0, 2)
        high_features = torch.cat([coarse_features, high_features], dim=0)

        class_embedding = None
    
        T_cls = []
        loss_part = []
        prediction_sum = 0
        loss_sum = 0
        for j in range(self.part_num):
            high_feature = high_features[j]
            patch_features = None
            total_loss = 0
            prediction = []
            if j>0:
                mask_p = mask[:, j-1].unsqueeze(1)
            cls_tokens = self.cls_token[j][0].expand(-1, bs, -1)
            additional_features = high_feature
            patch_features = torch.cat((cls_tokens, additional_features), dim=0)
            patch_features = self.norm[j](patch_features)
            patch_features = self.transformer[j](patch_features)
            cls_tokens = patch_features[:3]
            
            # Visual-only logits: concatenate 3 CLS tokens and classify
            vis_feat = cls_tokens.permute(1, 0, 2).reshape(cls_tokens.shape[1], -1)
            logits = self.classifier(vis_feat)
            prediction_ = logits
            probs = torch.softmax(logits, dim=1)
            loss = self.cels[0](probs, label_index)
            
            if j>0:
                prediction_ = prediction_ * mask_p
                loss = loss * mask_p
            loss = loss.sum()
            total_loss = total_loss + (loss * self.loss_weights[-1])
            prediction = self.logsoftmax(prediction_)
            pred_classes = torch.argmax(prediction, dim=1)
            accuracy = (pred_classes == label_index).sum().item()
            loss_sum = total_loss + loss_sum
            loss_part.append(total_loss)
            prediction_sum = prediction_sum + prediction
            T_cls.append(accuracy)

        pred_classes = torch.argmax(prediction_sum, dim=1)
        final_accuracy = (pred_classes == label_index).sum().item()
        T_cls.append(final_accuracy)
        loss_part.append(loss_sum / self.part_num)

        return loss_part, T_cls