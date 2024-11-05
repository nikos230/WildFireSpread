import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.dataset import BurnedAreaDataset
from unet.model_new import UNet3D
from unet.loss import FocalLoss, IoULoss, BCEIoULoss, BCEDiceLoss, F1ScoreLoss
from utils.utils import dice_coefficient
from utils.utils import f1_score
from utils.utils import accuracy
from utils.utils import iou
from utils.utils import recall
from utils.utils import auroc
from utils.utils import load_files
import os
import numpy as np
import yaml
import ast
import wandb


def train(dataset_path, checkpoints, num_filters, kernel_size, pool_size, use_batchnorm, final_activation, num_epochs, batch_size, learing_rate, drop_out_rate, train_years, validation_years):
    # make checkpoints folder if not exist
    os.makedirs(checkpoints, exist_ok=True)

    # load train and validation files from folders | Dataset and Dataloader
    train_files = load_files(dataset_path, train_years)
    validation_files = load_files(dataset_path, validation_years)
    
    train_dataset = BurnedAreaDataset(train_files)
    validation_dataset = BurnedAreaDataset(validation_files)

    print(f'Number of train samples: {len(train_dataset)}\nNumber of validation samples: {len(validation_dataset)}')
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print("Batch index:", batch_idx)
        print("Input shape:", inputs.shape)   # Shape of the input tensor
        print("Label shape:", labels.shape)   # Shape of the label tensor
        break  # Only check the shape of the first batch
    #exit()

    # model, loss, optimizer
    # set input channels (nunber of dynamic + static variables)
    input_channels = train_dataset[0][0].shape[0] # get input from input_tensor
    output_channels = 1 # binary classification

    model = UNet3D(
        in_channels=input_channels,
        out_channels=output_channels, 
        num_filters=num_filters, 
        kernel_size=kernel_size, 
        pool_size=pool_size, 
        use_batchnorm=use_batchnorm, 
        final_activation=final_activation,
        dropout_rate=drop_out_rate
        )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(alpha=0.8, gamma=0.5)
    #criterion = IoULoss()
    #criterion = BCEDiceLoss()
    #criterion = F1ScoreLoss()
    criterion = BCEIoULoss()
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)#, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learing_rate, weight_decay=1e-2)  # You can tune weight decay

    for epoch in range(num_epochs):
        model.train()
        epoch_loss     = 0
        epoch_dice     = 0
        epoch_f1       = 0
        #epoch_accuracy = 0
        epoch_iou      = 0
        epoch_recall   = 0
        epoch_auroc    = 0

        for inputs, labels in train_loader:
            #print(inputs[0])
            #exit()
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1) # add a dimention 
      
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
            epoch_auroc += auroc(outputs, labels).item()

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_f1   = epoch_f1   / len(train_loader)
        #avg_accuracy = epoch_accuracy / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_recall = epoch_recall / len(train_loader)
        avg_auroc = epoch_auroc / len(train_loader)
        #wandb.log({"Train Loss": avg_loss, "Train Dice Coefficient": avg_dice, "Train F1 Score": avg_f1,
                   #"Train IoU": avg_iou, "Train Recall": avg_recall, "Train AUROC": avg_auroc, "epoch": epoch})
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}, f1 Score: {avg_f1:.4f}, iou: {avg_iou:.4f}, auroc: {avg_auroc:.4f}')
        
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
        validation_auroc = 0
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
                validation_auroc += auroc(outputs, labels).item()

        avg_validation_loss = validation_loss / len(validation_loader)
        avg_validation_dice = validation_dice / len(validation_loader)
        avg_validation_f1   = validation_f1   / len(validation_loader)
        #avg_validation_accuracy = validation_accuracy / len(validation_loader)
        avg_validation_iou = validation_iou / len(validation_loader)
        avg_validation_recall = validation_recall / len(validation_loader)
        avg_validation_auroc = validation_auroc / len(validation_loader)
        #wandb.log({"Validation Loss": avg_validation_loss, "Validation Dice Coefficient": avg_validation_dice, 
                   #"Validation F1 Score": avg_validation_f1, "Validation IoU": avg_validation_iou, 
                   #"Validation Recall": avg_validation_recall, "Validation AUROC": avg_validation_auroc, "epoch": epoch})
        print(f'Validation Loss: {avg_validation_loss:.4f}, Validation Dice Coefficient: {avg_validation_dice:.4f}, f1 Score: {avg_validation_f1:.4f}, iou: {avg_validation_iou:.4f}, auroc: {avg_validation_auroc:.4f}\n')
        

if __name__ == '__main__':

    with open('WildFireSpread/WildFireSpread_UNET/configs/train_test_config.yaml', 'r') as t_config:
        train_config = yaml.safe_load(t_config)
    t_config.close()

    with open('WildFireSpread/WildFireSpread_UNET/configs/dataset.yaml', 'r') as d_config:
        dataset_config = yaml.safe_load(d_config)
    d_config.close()    

    #wandb.init(project="WildFireSpread", config=train_config)

    dataset_path     = dataset_config['dataset']['corrected_dataset_path']
    validation_years = dataset_config['samples']['validation_years']
    test_years       = dataset_config['samples']['test_years']
    train_years      = dataset_config['samples']['train_years']
    checkpoints      = train_config['model']['checkpoints']
    num_filters      = train_config['model']['num_filters']
    kernel_size      = train_config['model']['kernel_size']
    pool_size        = train_config['model']['pool_size']
    use_batchnorm    = train_config['model']['use_batchnorm']
    final_activation = train_config['model']['final_activation']
    num_epochs       = train_config['training']['number_of_epochs']
    batch_size       = train_config['training']['batch_size']
    learing_rate     = train_config['training']['learing_rate']
    drop_out_rate    = train_config['training']['drop_out_rate']

    if train_years == 'all':
        # find all avaible years, exclude validation and test years and use the rest for training
        all_years = os.listdir(dataset_path)
        train_years = [year for year in all_years if year not in [validation_years, test_years]]
    else:
        train_years = [train_years]
        train_years = train_years[0].split(', ')

    validation_years = [validation_years]
    validation_years = validation_years[0].split(', ')
    

    # Log config and model to wandb
    #wandb.config.update({"dataset_path": dataset_path, "checkpoints": checkpoints, "num_filters": num_filters,
                         #"kernel_size": kernel_size, "pool_size": pool_size, "use_batchnorm": use_batchnorm,
                         #"final_activation": final_activation, "num_epochs": num_epochs, "batch_size": batch_size,
                         #"learning_rate": learing_rate, "drop_out_rate": drop_out_rate})

    print(f'Currect settings for traing: \n Train dataset path: {dataset_path} \n Checkpoints save path: {checkpoints} \n Number of filters: {num_filters} \n Kernel Size: {kernel_size} \n Pool Size: {pool_size} \n Use Batchnoorm: {use_batchnorm} \n Final Activation: {final_activation} \n Number of Epochs: {num_epochs} \n Batch Size: {batch_size} \n Learing Rate: {learing_rate} \n Drop out Rate: {drop_out_rate} \n')
    
    train(dataset_path, checkpoints, ast.literal_eval(num_filters), ast.literal_eval(kernel_size), ast.literal_eval(pool_size), bool(use_batchnorm), ast.literal_eval(final_activation), int(num_epochs), int(batch_size), float(learing_rate), float(drop_out_rate), train_years, validation_years)