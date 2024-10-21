import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.dataset import BurnedAreaDataset
from unet.model_new import UNet3D
from unet.loss import FocalLoss, IoULoss, BCEIoULoss
from utils.utils import dice_coefficient
from utils.utils import f1_score
from utils.utils import accuracy
from utils.utils import iou
from utils.utils import recall
import glob
import os
import numpy as np
import yaml
import ast

def split_dataset(dataset, validation_size=0.2, test_size=0.1):
    indices = list(range(len(dataset)))
    train_validation_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, validation_indices = train_test_split(train_validation_indices, test_size=validation_size, random_state=42)

    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, validation_set, test_set


def train(dataset_path, checkpoints, num_filters, kernel_size, pool_size, use_batchnorm, final_activation, num_epochs, batch_size, learing_rate):

    # make checkpoints folder if not exist
    os.makedirs(checkpoints, exist_ok=True)
    # Dataset and Dataloader
    nc_files = glob.glob(dataset_path + '/*.nc')
    dataset = BurnedAreaDataset(nc_files)

    print(f'Number of samples: {len(nc_files)}')

    # split the dataset into train, validation, test sets
    train_set, validation_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model, loss, optimizer
    # set input channels (nunber of dynamic + static variables)
    input_channels = dataset[0][0].shape[0] # get input from input_tensor
    output_channels = 1 # binary classification

    model = UNet3D(
        in_channels=input_channels,
        out_channels=output_channels, 
        num_filters=num_filters, 
        kernel_size=kernel_size, 
        pool_size=ast.literal_eval(pool_size), 
        use_batchnorm=use_batchnorm, 
        final_activation=final_activation,
        dropout_rate=0.3)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(alpha=0.8, gamma=0.5)
    criterion = IoULoss()
    #criterion = BCEIoULoss()
    optimizer = optim.Adam(model.parameters(), lr=learing_rate, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learing_rate, weight_decay=1e-2)  # You can tune weight decay

    for epoch in range(num_epochs):
        model.train()
        epoch_loss     = 0
        epoch_dice     = 0
        epoch_f1       = 0
        #epoch_accuracy = 0
        epoch_iou      = 0
        epoch_recall   = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1) # add a dimention 
            #print('Label size after unsqueeze:', labels.shape)
            #print('Input size:', inputs.shape)
      
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(outputs, labels).item()
            epoch_f1   += f1_score(outputs, labels).item()
            #epoch_accuracy += accuracy(outputs, labels).item()
            epoch_iou += iou(outputs, labels).item()
            epoch_recall += recall(outputs, labels).item()

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_f1   = epoch_f1   / len(train_loader)
        #avg_accuracy = epoch_accuracy / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_recall = epoch_recall / len(train_loader)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}, f1 Score: {avg_f1:.4f}, iou: {avg_iou:.4f}, recall: {avg_recall:.4f}')

        # save model chackpoint
        checkpoint_path = os.path.join(checkpoints, f'model_epoch{epoch+1}.pth')
        #torch.save(model.state_dict(), checkpoint_path)

        # validate the model on validation set
        model.eval()
        validation_loss = 0
        validation_dice = 0
        validation_f1   = 0
        #validation_accuracy = 0
        validation_iou = 0
        validation_recall = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                validation_dice += dice_coefficient(outputs, labels).item()
                validation_f1   += f1_score(outputs, labels).item()
                #validation_accuracy += accuracy(outputs, labels).item()
                validation_iou += iou(outputs, labels).item()
                validation_recall += recall(outputs, labels).item()

        avg_validation_loss = validation_loss / len(validation_loader)
        avg_validation_dice = validation_dice / len(validation_loader)
        avg_validation_f1   = validation_f1   / len(validation_loader)
        #avg_validation_accuracy = validation_accuracy / len(validation_loader)
        avg_validation_iou = validation_iou / len(validation_loader)
        avg_validation_recall = validation_recall / len(validation_loader)
        print(f'Validation Loss: {avg_validation_loss:.4f}, Validation Dice Coefficient: {avg_validation_dice:.4f}, f1 Score: {avg_validation_f1:.4f}, iou: {avg_validation_iou:.4f}, recall: {avg_validation_recall:.4f}\n')
        

if __name__ == '__main__':

    with open('WildFireSpread/WildFireSpread_UNET/configs/train_config.yaml', 'r') as train_config:
        config = yaml.safe_load(train_config)
    train_config.close()

    dataset_path     = config['dataset']['train_dataset']
    checkpoints      = config['dataset']['checkpoints']
    num_filters      = config['model']['num_filters']
    kernel_size      = config['model']['kernel_size']
    pool_size        = config['model']['pool_size']
    use_batchnorm    = config['model']['use_batchnorm']
    final_activation = config['model']['final_activation']
    num_epochs       = config['training']['number_of_epochs']
    batch_size       = config['training']['batch_size']
    learing_rate     = config['training']['learing_rate']

    print(f'Currect settings for traing: \n Train dataset path: {dataset_path} \n Checkpoints save path: {checkpoints} \n Number of filters: {num_filters} \n Kernel Size: {kernel_size} \n Pool Size: {pool_size} \n Use Batchnoorm: {use_batchnorm} \n Final Activation: {final_activation} \n')
    print(f'More training settings: \n Number of Epochs: {num_epochs} \n Batch Size: {batch_size} \n Learing Rate: {learing_rate} \n')
    train(dataset_path, checkpoints, num_filters, kernel_size, pool_size, use_batchnorm, final_activation, num_epochs, batch_size, float(learing_rate))