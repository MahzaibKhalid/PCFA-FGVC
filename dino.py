import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

class DINO(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        self.patch_size = self.model.config.patch_size
        
    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)
        return outputs.last_hidden_state
    
    def get_feat_maps(self, x):
        """Extract feature maps from DINOv2"""
        # Get the patch embeddings
        with torch.no_grad():
            outputs = self.model(x)
            hidden_states = outputs.last_hidden_state
            
            # Remove CLS token
            patch_embeddings = hidden_states[:, 1:, :]
            
            # Reshape to feature map format (B, C, H, W)
            batch_size = x.shape[0]
            h = w = int((patch_embeddings.shape[1]) ** 0.5)
            c = patch_embeddings.shape[2]
            
            feat_maps = patch_embeddings.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
            
        return feat_maps 