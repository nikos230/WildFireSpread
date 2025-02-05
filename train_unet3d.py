import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.dataset_unet3d import BurnedAreaDataset
from unet.model_unet3d_struct import UNet3D_struct
from unet.model_unet3d import UNet3D
from unet.loss import FocalLoss, IoULoss, BCEIoULoss, BCEDiceLoss, F1ScoreLoss
from utils.utils import dice_coefficient,  f1_score, accuracy, iou, recall, auroc, precision, load_files, load_files_train, load_files_train_, load_files_validation
import os
import numpy as np
import yaml
import ast
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def train(dataset_path, checkpoints, num_filters, kernel_size, pool_size, use_batchnorm, final_activation, num_epochs, batch_size, learing_rate, drop_out_rate, train_years, validation_years, threshold, num_layers, train_countries, val_countries, burned_area_big, burned_area_ratio):
    # make checkpoints folder if not exist
    os.makedirs(checkpoints, exist_ok=True)
    
    batch_size = 28

    # load train and validation files from folders | Dataset and Dataloader
    if burned_area_big == 0 and burned_area_ratio == 0:
        train_files = load_files_train(dataset_path, train_years, train_countries)
    else:
        train_files = load_files_train_(dataset_path, train_years, train_countries, burned_area_big, burned_area_ratio)


    validation_files = load_files_validation(dataset_path, validation_years, val_countries)
    
    train_dataset = BurnedAreaDataset(train_files)
    validation_dataset = BurnedAreaDataset(validation_files)

    print(f'Number of train samples: {len(train_dataset)}\nNumber of validation samples: {len(validation_dataset)}')
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print("Batch index:", batch_idx)
        print("Input shape:", inputs.shape)   # input tensor
        print("Label shape:", labels.shape)   # label tensor
        break  
    

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
        dropout_rate=drop_out_rate,
        num_layers=num_layers
        )

    # model = UNet3D_struct(
    #     in_channels=input_channels,
    #     out_channels=output_channels, 
    #     )  
         

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model)
    print(device)
    model.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(alpha=0.9, gamma=1)
    #criterion = IoULoss()
    criterion = BCEDiceLoss()
    #criterion = F1ScoreLoss()
    #criterion = BCEIoULoss()
    optimizer = optim.Adam(model.parameters(), lr=learing_rate, weight_decay=1e-4)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learing_rate, weight_decay=1e-2)  # You can tune weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss      = 0
        epoch_dice      = 0
        epoch_f1        = 0
        epoch_accuracy  = 0
        epoch_iou       = 0
        epoch_recall    = 0
        epoch_auroc     = 0
        epoch_precision = 0

        for inputs, labels in tqdm(train_loader):
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
            epoch_dice += dice_coefficient(outputs, labels, threshold).item()
            epoch_f1   += f1_score(outputs, labels, threshold).item()
            epoch_accuracy += accuracy(outputs, labels, threshold).item()
            epoch_iou += iou(outputs, labels, threshold).item()
            epoch_recall += recall(outputs, labels, threshold).item()
            epoch_auroc += auroc(outputs, labels).item()
            epoch_precision += precision(outputs, labels, threshold).item()


        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_f1   = epoch_f1   / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_recall = epoch_recall / len(train_loader)
        avg_auroc = epoch_auroc / len(train_loader)
        avg_precision = epoch_precision / len(train_loader)

        wandb.log({"Train Loss": avg_loss, "Train Dice Coefficient": avg_dice, "Train F1 Score": avg_f1,
                   "Train IoU": avg_iou, "Train Recall": avg_recall, "Train AUROC": avg_auroc, "Train Precision": avg_precision, "Train Accuracy": avg_accuracy, "epoch": epoch+1})
                   
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}, f1 Score: {avg_f1:.4f}, iou: {avg_iou:.4f}, auroc: {avg_auroc:.4f}')
        
        # save model chackpoint
        checkpoint_path = os.path.join(checkpoints, f'model_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # validate the model on validation set
        model.eval()
        validation_loss = 0
        validation_dice = 0
        validation_f1   = 0
        validation_accuracy = 0
        validation_iou = 0
        validation_recall = 0
        validation_auroc = 0
        validation_precision = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                validation_dice += dice_coefficient(outputs, labels, threshold).item()
                validation_f1   += f1_score(outputs, labels, threshold).item()
                validation_accuracy += accuracy(outputs, labels, threshold).item()
                validation_iou += iou(outputs, labels, threshold).item()
                validation_recall += recall(outputs, labels, threshold).item()
                validation_auroc += auroc(outputs, labels).item()
                validation_precision += precision(outputs, labels, threshold).item()

        avg_validation_loss = validation_loss / len(validation_loader)
        avg_validation_dice = validation_dice / len(validation_loader)
        avg_validation_f1   = validation_f1   / len(validation_loader)
        avg_validation_accuracy = validation_accuracy / len(validation_loader)
        avg_validation_iou = validation_iou / len(validation_loader)
        avg_validation_recall = validation_recall / len(validation_loader)
        avg_validation_auroc = validation_auroc / len(validation_loader)
        avg_validation_precision = validation_precision / len(validation_loader)

        scheduler.step(avg_validation_loss)

        wandb.log({"Validation Loss": avg_validation_loss, "Validation Dice Coefficient": avg_validation_dice, 
                   "Validation F1 Score": avg_validation_f1, "Validation IoU": avg_validation_iou, 
                   "Validation Recall": avg_validation_recall, "Validation AUROC": avg_validation_auroc, "Validation Accuracy": avg_accuracy, "Validation Precision": avg_validation_precision, "epoch": epoch+1})

        print(f'Validation Loss: {avg_validation_loss:.4f}, Validation Dice Coefficient: {avg_validation_dice:.4f}, f1 Score: {avg_validation_f1:.4f}, iou: {avg_validation_iou:.4f}, auroc: {avg_validation_auroc:.4f}\n')
        

if __name__ == '__main__':
    os.system("clear")

    with open('configs/train_test_config_unet3d.yaml', 'r') as t_config:
        train_config = yaml.safe_load(t_config)
    t_config.close()

    with open('configs/dataset.yaml', 'r') as d_config:
        dataset_config = yaml.safe_load(d_config)
    d_config.close()

    with open('configs/available_countries.yaml', 'r') as c_config:
        countries_config = yaml.safe_load(c_config)
    c_config.close()
    
    wandb.init(project="WildFireSpread", config=train_config)

    dataset_path      = dataset_config['dataset']['corrected_dataset_path']
    validation_years  = dataset_config['samples']['validation_years']
    test_years        = dataset_config['samples']['test_years']
    train_years       = dataset_config['samples']['train_years']
    train_countries   = dataset_config['samples']['train_countries']
    val_countries     = dataset_config['samples']['validation_countries']
    test_countries    = dataset_config['samples']['test_countries']
    ex_count_train    = dataset_config['samples']['exclude_countries_from_train']
    ex_count_val      = dataset_config['samples']['exclude_countries_from_val']
    burned_area_big   = dataset_config['samples']['bunred_area_bigger_than']
    burned_area_ratio = 'None'#dataset_config['samples']['burned_area_ratio']
    checkpoints       = train_config['model']['checkpoints']
    num_filters       = train_config['model']['num_filters']
    kernel_size       = train_config['model']['kernel_size']
    pool_size         = train_config['model']['pool_size']
    use_batchnorm     = train_config['model']['use_batchnorm']
    final_activation  = train_config['model']['final_activation']
    num_layers        = train_config['model']['num_layers']
    threshold         = train_config['model']['threshold']
    drop_out_rate     = train_config['model']['drop_out_rate']
    num_epochs        = train_config['training']['number_of_epochs']
    batch_size        = train_config['training']['batch_size']
    learing_rate      = train_config['training']['learing_rate']

    # find all available years, exclude validation and test years and use the rest for training
    if train_years == 'all':
        all_years = os.listdir(dataset_path)
        train_years = [year for year in all_years if year not in [validation_years, test_years]]
    else:
        train_years = [train_years]
        train_years = train_years[0].split(', ')

    validation_years = [validation_years]
    validation_years = validation_years[0].split(', ')
    
    # get list of available countries in the dataset
    all_countries = []
    for country, value in countries_config['countries'].items():
        if value == 'True':
            all_countries.append(country)
        else:
            print(f'Country: {country} not found or disabled in available_countries.yaml')    

    # find train countries and exclude some if defined in dataset.yaml
    if train_countries == 'all':
        train_countries = all_countries
    else:
        train_countries = [train_countries]
        train_countries = train_countries[0].split(', ')

    if val_countries == 'all':
        val_countries = all_countries
    else:
        val_countries = [val_countries]
        val_countries = val_countries[0].split(', ')    

    if ex_count_train != '':
        ex_count_train = [ex_count_train]
        ex_count_train = ex_count_train[0].split(', ')
        train_countries = list(set(train_countries) - set(ex_count_train))   

    if ex_count_val != '':
        ex_count_val = [ex_count_val]
        ex_count_val = ex_count_val[0].split(', ')
        val_countries = list(set(val_countries) - set(ex_count_val))        

    if burned_area_big == 'None':
        burned_area_big = 0
    if burned_area_ratio == 'None':
        burned_area_ratio = 0    

    # config and model to wandb
    wandb.config.update({"dataset_path": dataset_path, "checkpoints": checkpoints, "num_filters": num_filters,
                         "kernel_size": kernel_size, "pool_size": pool_size, "use_batchnorm": use_batchnorm,
                         "final_activation": final_activation, "num_epochs": num_epochs, "batch_size": batch_size,
                         "learning_rate": learing_rate, "drop_out_rate": drop_out_rate})

    print(f'Currect settings for traing: \n Train dataset path: {dataset_path} \n Checkpoints save path: {checkpoints} \n Number of filters: {num_filters} \n Kernel Size: {kernel_size} \n Pool Size: {pool_size} \n Use Batchnoorm: {use_batchnorm} \n Final Activation: {final_activation} \n Number of Epochs: {num_epochs} \n Batch Size: {batch_size} \n Learing Rate: {learing_rate} \n Drop out Rate: {drop_out_rate} \n Threshold: {threshold} \n Number of Layers (ConvBlock): {num_layers} \n')
    
    train(dataset_path, checkpoints, ast.literal_eval(num_filters), ast.literal_eval(kernel_size), ast.literal_eval(pool_size), bool(use_batchnorm), ast.literal_eval(final_activation), int(num_epochs), int(batch_size), float(learing_rate), float(drop_out_rate), train_years, validation_years, float(threshold), int(num_layers), train_countries, val_countries, int(burned_area_big), float(burned_area_ratio))