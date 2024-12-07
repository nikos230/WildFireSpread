import numpy as np
import torch
import torchmetrics
import os
import glob
import xarray as xr

def normalize_data(data):
    # Normalize data to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return data


def load_files_train(dataset_path, years, countries):
    # load train or validation set
    files = []
    for year in years:
        year_path = os.path.join(dataset_path, year)
        for country in countries:
            country_path = os.path.join(year_path, country)
            new_files = glob.glob(country_path + '/*.nc')
            
            files.append(new_files)

    files = [item for sublist in files for item in sublist]

    return files


def load_files_train_(dataset_path, years, countries, burned_area_big, burned_area_ratio):
    # load train or validation set
    files = []
    for year in years:
        year_path = os.path.join(dataset_path, year)
        for country in countries:
            country_path = os.path.join(year_path, country)
            new_files = glob.glob(country_path + '/*.nc')
            for new_file in new_files:
                path_to_file = os.path.join(country_path, new_file)

                ds = xr.open_dataset(path_to_file)
                
                if ds.attrs['burned_area_ha'] < burned_area_big:
                    continue
                #if int(ds.attrs['date'].split('-')[1]) >= 6 and int(ds.attrs['date'].split('-')[1]) <= 8:
                ds.close()
                files.append(new_file)

    #files = [item for sublist in files for item in sublist]

    return files    


def load_files_validation(dataset_path, years, countries):
    # load train or validation set
    files = []
    for year in years:
        year_path = os.path.join(dataset_path, year)
        for country in countries:
            country_path = os.path.join(year_path, country)
            new_files = glob.glob(country_path + '/*.nc')
            
            files.append(new_files)

    files = [item for sublist in files for item in sublist]

    return files   



def load_files(dataset_path, years, countries):
    # load train or validation set
    files = []
    for year in years:
        year_path = os.path.join(dataset_path, year)
        for country in countries:
            country_path = os.path.join(year_path, country)
            new_files = glob.glob(country_path + '/*.nc')
            
            files.append(new_files)

    files = [item for sublist in files for item in sublist]

    return files


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


def precision(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Compute the precision metric.

    Args:
        preds (torch.Tensor): Predicted logits or probabilities.
        targets (torch.Tensor): Ground truth binary labels.
        threshold (float): Threshold for converting probabilities to binary predictions. Defaults to 0.5.
        smooth (float): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        float: The computed precision value.
    """
    preds = torch.sigmoid(preds)  # Apply sigmoid activation to convert logits to probabilities
    preds = (preds > threshold).float()  # Convert probabilities to binary predictions
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().float()  # True positives
    fp = (preds * (1 - targets)).sum().float()  # False positives

    precision_value = (tp + smooth) / (tp + fp + smooth)  # Precision calculation
    return precision_value    

    