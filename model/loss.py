import torch
import torch.nn as nn
import numpy as np

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon, gamma=4.0):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon[0]
        self.focal_weight_epsilon_t = epsilon[1]
        self.gamma = gamma
        
    def forward(self, probs, targets):
        smooth_targets = torch.index_select(self.epsilon, 0, targets)
        focal_weight_targets_t = torch.index_select(self.focal_weight_epsilon_t, 0, targets)  
        probs_pt = probs * focal_weight_targets_t
        pt = probs_pt.mean(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        log_probs = torch.log(probs + 1e-12)
        loss = -focal_weight.unsqueeze(1) * smooth_targets * log_probs
        loss = loss
        return loss
    