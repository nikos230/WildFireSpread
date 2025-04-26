
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        """
        Dice Loss for binary segmentation tasks.
        
        Parameters:
        - smooth: Smoothing factor to avoid division by zero (default is 1e-6).
        - reduction: 'mean' or 'sum' to apply reduction over batch.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Parameters:
        - preds: Predictions (logits before sigmoid), shape: (batch_size, 1, height, width, ...)
        - targets: Ground truth labels (binary 0 or 1), shape: (batch_size, 1, height, width, ...)
        """
        # Apply sigmoid to convert logits to probabilities
        preds = torch.sigmoid(preds)

        # Flatten the tensors to compute dice over the entire volume
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (preds * targets).sum()
        total = preds.sum() + targets.sum()

        # Compute Dice score and Dice loss
        dice_score = (2.0 * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1 - dice_score

        # Apply reduction ('mean' or 'sum')
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class F1ScoreLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        """
        F1 Score Loss for binary classification tasks. Minimizing this loss is equivalent to maximizing the F1 score.
        
        Parameters:
        - smooth: Smoothing factor to avoid division by zero (default is 1e-6).
        - reduction: 'mean' or 'sum' to apply reduction over batch.
        """
        super(F1ScoreLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Parameters:
        - preds: Predictions (logits before sigmoid), shape: (batch_size, 1, height, width, ...)
        - targets: Ground truth labels (binary 0 or 1), shape: (batch_size, 1, height, width, ...)
        """
        # Apply sigmoid to convert logits to probabilities
        preds = torch.sigmoid(preds)

        # Flatten the predictions and targets to compute metrics over all pixels
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Compute precision and recall components
        TP = (preds * targets).sum()  # True Positives
        FP = ((1 - targets) * preds).sum()  # False Positives
        FN = (targets * (1 - preds)).sum()  # False Negatives

        # Compute precision and recall
        precision = (TP + self.smooth) / (TP + FP + self.smooth)
        recall = (TP + self.smooth) / (TP + FN + self.smooth)

        # Compute F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + self.smooth)

        # F1 loss is 1 - F1 score
        f1_loss = 1 - f1_score

        # Apply reduction ('mean' or 'sum')
        if self.reduction == 'mean':
            return f1_loss.mean()
        elif self.reduction == 'sum':
            return f1_loss.sum()
        else:
            return f1_loss





class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


