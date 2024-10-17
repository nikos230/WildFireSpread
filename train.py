# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.dataset import BurnedAreaDataset
from unet.model import UNet3D
from utils.utils import dice_coefficient
import glob
import os
import numpy as np

def split_dataset(dataset, val_size=0.2, test_size=0.1):
    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=42)
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set

def train():
    # Hyperparameters
    num_epochs = 20
    batch_size = 2
    learning_rate = 1e-4

    # Paths
    data_path = 'WildFireSpread/dataset_small_corrected/*.nc'  # Update with your data path
    checkpoint_dir = 'WildFireSpread/WildFireSpread_UNET/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    nc_files = glob.glob(data_path)
    dataset = BurnedAreaDataset(nc_files)
    

    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    in_channels = dataset[0][0].shape[0]  # Number of input channels
    out_channels = 1  # Binary segmentation
    model = UNet3D(in_channels, out_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.unsqueeze(1)  # Add channel dimension

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(outputs, targets).item()

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}')

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # Validate on validation set
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1)  # Add channel dimension

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Dice Coefficient: {avg_val_dice:.4f}')

if __name__ == '__main__':
    train()
