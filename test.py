import torch
from torch.utils.data import DataLoader
from unet.model import UNet
from utils.dataset import NetCDFDataset
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import sys
import os

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = f1_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    iou = jaccard_score(y_true_flat, y_pred_flat)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }

# load model from checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=18 * 5, out_channels=1)
model.load_state_dict(torch.load('saved_models/unet_model_test.pth'))
model.to(device)

# load test dataset
TEST_FILES_PATH = 'WildFireSpread/dataset_small'
def get_nc_files(directory):
    nc_file_paths = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.nc'):
                nc_file_paths.append(entry.path)
    return nc_file_paths
test_files = get_nc_files(TEST_FILES_PATH)
test_dataset = NetCDFDataset(test_files, split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# test the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# calculate metrics
metrics = compute_metrics(all_labels, all_preds)
print('Test metrics:')
for metric_name, metric_value in metrics.items():
    print(f'{metric_name.capitalize()}: {metric_value:.2f}')




def visualize_results(inputs, labels, preds, num_samples=5):
    """
    Visualize the results of predictions.
    
    Args:
    - inputs (np.ndarray): Input samples.
    - labels (np.ndarray): Ground truth labels.
    - preds (np.ndarray): Predicted labels.
    - num_samples (int): Number of samples to visualize.
    """
    num_samples = min(num_samples, len(inputs))
    
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        # Plot input
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(inputs[i][0], cmap='gray')  # Display the first channel
        plt.title("Input")
        plt.axis("off")

        # Plot ground truth
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(labels[i][0], cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        # Plot predictions
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds[i][0], cmap='gray')
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Visualize the results
visualize_results(all_inputs, all_labels, all_preds)