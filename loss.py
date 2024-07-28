import torch
import torch.nn as nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)): 
                alpha_tensor = torch.full_like(inputs, self.alpha)
            else:  
                alpha_tensor = torch.zeros_like(inputs)
                alpha_tensor.scatter_(1, targets.view(-1, 1), self.alpha)
            focal_loss = alpha_tensor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        else:
            return focal_loss
        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        
        outputs_flat = outputs.view(outputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = torch.sum(outputs_flat * targets_flat, dim=1)
        union = torch.sum(outputs_flat, dim=1) + torch.sum(targets_flat, dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        
        elif self.reduction == 'sum':
            return loss.sum()
        
        else:
            return loss

    

