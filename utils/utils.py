import numpy as np
import torch
import torchmetrics

def normalize_data(data):
    # Normalize data to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return data

def dice_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def f1_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)  # Apply sigmoid activation to convert logits to probabilities
    preds = (preds > threshold).float()  # Convert probabilities to binary predictions
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().float()  # True positives
    fp = (preds * (1 - targets)).sum().float()  # False positives
    fn = ((1 - preds) * targets).sum().float()  # False negatives

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1

def accuracy(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)  # Apply sigmoid to convert logits to probabilities
    preds = (preds > threshold).float()  # Convert probabilities to binary predictions
    correct = (preds == targets).float()  # Compare predictions to actual targets
    accuracy = correct.sum() / len(correct)  # Calculate accuracy as mean of correct predictions
    return accuracy

def iou(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)  # Apply sigmoid activation to convert logits to probabilities
    preds = (preds > threshold).float()  # Convert probabilities to binary predictions
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum().float()
    union = preds.sum() + targets.sum() - intersection

    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def recall(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)  # Apply sigmoid activation to convert logits to probabilities
    preds = (preds > threshold).float()  # Convert probabilities to binary predictions
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().float()  # True positives
    fn = ((1 - preds) * targets).sum().float()  # False negatives

    recall_value = (tp + smooth) / (tp + fn + smooth)  # Recall calculation
    return recall_value


def auroc(preds, targets, num_classes=2):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC).

    Args:
        preds (torch.Tensor): Predicted logits or probabilities.
        targets (torch.Tensor): Ground truth binary labels.
        num_classes (int): Number of classes. Defaults to 2 (binary classification).

    Returns:
        auroc_value: The computed AUROC value.
    """
    preds = torch.sigmoid(preds)  # Apply sigmoid to convert logits to probabilities

    # Set the appropriate task for AUROC
    if num_classes == 2:
        task = "binary"
    else:
        task = "multiclass"

    # Initialize AUROC metric
    auroc_metric = torchmetrics.AUROC(task=task, num_classes=num_classes)
    auroc_value = auroc_metric(preds, targets.int())  # Compute AUROC

    return auroc_value

    