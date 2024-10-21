
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Parameters:
        - alpha: Weighting factor for the class imbalance (default is 0.25, set to None for no weighting)
        - gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted (default is 2.0)
        - reduction: 'mean' or 'sum' to apply reduction over batch
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Parameters:
        - logits: raw model outputs (before sigmoid), shape: (batch_size, 1, ...)
        - targets: ground truth labels (0 or 1), shape: (batch_size, 1, ...)
        """
        # Sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid log(0) errors

        # Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute the focal loss factor
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t for the true class
        focal_factor = (1 - pt) ** self.gamma

        # Apply alpha balancing
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_factor = alpha_factor * focal_factor

        # Compute the final focal loss
        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Apply sigmoid activation to get probabilities
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou  # IoU loss (1 - IoU for optimization)



class BCEIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BCEIoULoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Combines Sigmoid with BCE
        self.smooth = smooth

    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)
        
        # IoU Loss
        preds = torch.sigmoid(preds)  # Apply sigmoid activation
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou  # IoU loss (1 - IoU)

        # Combine BCE and IoU loss
        total_loss = bce_loss + iou_loss
        return total_loss


