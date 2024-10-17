import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.dataset import BurnedAreaDataset
from utils.model import UNet3D
from utils.utils import dice_coefficient
import glob
import os
import numpy as np


def split_dataset(dataset, validation_size=0.2, test_size=0.1):
    indices = list(range(len(dataset)))
    train_validation_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, validation_indices = train_test_split(train_validation_indices, test_size=validation_size, random_state=42)

    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, validation_set, test_set


def train():
    num_epochs = 20
    batch_size = 2
    learing_rate = 1e-4

    # paths for data
    data_path = 'WildFireSpread/dataset_small_corrected/*.nc'
    checkpoint_dir = 'WildFireSpread/WildFireSpread_UNET/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and Dataloader
    nc_files = glob.glob(data_path)
    dataset = BurnedAreaDataset(nc_files)

    # split the dataset into train, validation, test sets
    train_set, validation_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model, loss, optimizer
    # set input channels (nunber of dynamic + static variables)
    input_channels = dataset[0][0].shape[0] # get input from input_tensor
    output_channels = 1 # binary classification

    model = UNet3D(input_channels, output_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1) # add a dimention 
            print('Label size after unsqueeze:', labels.shape)
            print('Input size:', inputs.size)

            optimizer.zero_grad()
            ouputs = model(inputs)

            loss = criterion(ouputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(outputs, labels).items()

        avg_loss = epoch_loss / len(train_loader)
        avg_loss = epoch_dice / len(train_loader)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}')

        # save model chackpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # validate the model on validation set
        model.eval()
        validation_loss = 0
        validation_dice = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                





























if __name__ == '__main__':
    train()