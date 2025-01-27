import torch
from torch.utils.data import DataLoader
from utils.dataset_unet3d import BurnedAreaDataset
from unet.model_unet3d import UNet3D
from utils.utils import dice_coefficient, load_files, f1_score, accuracy, iou, recall, auroc, precision
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ast
import xarray as xr
from affine import Affine
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping
import fiona
from fiona.crs import from_epsg
import geopandas as gpd
import pandas as pd
from datetime import datetime
from shapely.geometry import shape
import imageio


def test(
        dataset_path, checkpoints,
        num_filters, kernel_size,
        pool_size, use_batchnorm,
        final_activation, drop_out_rate,
        train_years, validation_years, 
        threshold, num_layers, 
        checkpoint_path, test_countries,
        xAI_save_path):
    
    # load train and validation files from folders | Dataset and Dataloader
    test_files = load_files(dataset_path, test_years, test_countries)

    test_dataset = BurnedAreaDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Number of test samples: {len(test_loader)}')
    # Model
    input_channels = test_dataset[0][0].shape[0]
    output_channels = 1

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
        
    model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

 
    model.to(device)
    model.eval()
    
    total_dice = 0
    total_accuracy = 0
    total_f1_score = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    
    all_predictions = []
    all_ground_truths = []
    all_dice = []
    
    cnt = 0
    #with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.requires_grad_().to(device)
        targets = targets.to(device)
        targets = targets.unsqueeze(1)  # Add channel dimension

        inputs.retain_grad()

        outputs = model(inputs)

        outputs = outputs.mean()
        outputs.backward()

        dynamic_vars = [
            'd2m', 
            'ignition_points', 
            'lai', 
            'lst_day', 
            'lst_night',
            'ndvi', 
            'rh', 
            'smi', 
            'sp', 
            'ssrd',
            't2m', 
            'tp',
            #'wind_direction',
            #'wind_direction_sin',
            #'wind_direction_cos',
            'u',
            'v'
            #'wind_speed'
                ]

        static_vars = [
            'aspect',
            'curvature',
            'dem', 
            #'roads_distance',
            'slope',
            'lc_agriculture',
            'lc_forest', 
            'lc_grassland', 
            'lc_settlement',
            'lc_shrubland', 
            'lc_sparse_vegetation',
            'lc_water_bodies',
            'lc_wetland', 
            #'population'
        ]

        saliency = inputs.grad.abs()


        save_saliency_map(idx, saliency, dynamic_vars, static_vars, targets, inputs, xAI_save_path)

        if cnt == 2:
            print('Done maps for 3 samples')
            exit(2)
        cnt += 1    


        model.zero_grad()  # Clear model parameters' gradients
        inputs.grad = None
    
        # dice = dice_coefficient(outputs, targets, threshold=threshold).item()
        # accuracy_ = accuracy(outputs, targets, threshold=threshold).item()
        # f1_score_ = f1_score(outputs, targets, threshold=threshold).item()
        # iou_ = iou(outputs, targets, threshold=threshold).item()
        # precision_ = precision(outputs, targets, threshold=threshold).item()
        # recall_ = recall(outputs, targets, threshold=threshold).item()


        # total_dice += dice
        # total_accuracy += accuracy_
        # total_f1_score += f1_score_
        # total_iou += iou_
        # total_precision += precision_
        # total_recall += recall_

        # Apply sigmoid activation to get probabilities
        #preds = torch.sigmoid(outputs)
        #preds = preds.cpu().numpy()[0, 0]  # Shape: (height, width)
        #targets = targets.cpu().numpy()[0, 0]  # Shape: (height, width)

        # Optional: Threshold the predictions to get binary masks
        #binary_preds = (preds > threshold).astype(np.uint8)

        # Collect predictions and ground truths for visualization
        #all_predictions.append(binary_preds)
        #all_ground_truths.append(targets)
        #all_dice.append(dice)



    # avg_dice = total_dice / len(test_loader)
    # avg_accuracy = total_accuracy / len(test_loader)
    # avg_f1_score = total_f1_score / len(test_loader)
    # avg_iou = total_iou / len(test_loader)
    # avg_precision = total_precision / len(test_loader)
    # avg_recall = total_recall / len(test_loader)


    # print(f'Average Dice Coefficient on Test Set: {avg_dice:.4f}')
    # print(f'Average Accuracy Coefficient on Test Set: {avg_accuracy:.4f}')
    # print(f'Average f1 Score Coefficient on Test Set: {avg_f1_score:.4f}')
    # print(f'Average IoU Coefficient on Test Set: {avg_iou:.4f}')
    # print(f'Average Precision Coefficient on Test Set: {avg_precision:.4f}')
    # print(f'Average Recall Coefficient on Test Set: {avg_recall:.4f}')
 

   

def save_saliency_map(idx, saliency_maps, dynamic_vars, static_vars, targets, inputs, xAI_save_path):
    # saliency_maps shape [batch_size, channel, time_step, height, width]
    all_variables_names = dynamic_vars + static_vars
    path_to_maps = f'{xAI_save_path}/sample_{idx}'
    path_to_maps_nc = f'{xAI_save_path}/sample_{idx}_tif'

    os.makedirs(path_to_maps, exist_ok=True)
    os.makedirs(path_to_maps_nc, exist_ok=True)

    for time_step in range(0, saliency_maps.shape[2]):
        for variable in range(0, saliency_maps.shape[1]):
            day = 'day_' + str(time_step+1)
            path_to_map = os.path.join(path_to_maps, day)
            path_to_map_nc = os.path.join(path_to_maps_nc, day)
            os.makedirs(path_to_map, exist_ok=True)
            os.makedirs(path_to_map_nc, exist_ok=True)

            variable_saliency = saliency_maps[0, variable, time_step].cpu().detach().numpy()

            # save netcdf saliency map
            imageio.imwrite(f'{path_to_map_nc}/saliency_map_{all_variables_names[variable]}.tif', variable_saliency)


            variable_image = inputs[0, variable, time_step].cpu().detach().numpy()
            plt.imshow(variable_image, cmap='gray')

            #label = targets[0, 0].cpu().detach().numpy()
            #plt.imshow(label, cmap='gray')

            plt.imshow(variable_saliency, cmap='hot', alpha=0.3)
            plt.title(f'saliency map for variable {all_variables_names[variable]}')
            cbar = plt.colorbar()
            plt.savefig(f'{path_to_map}/saliency_map_{all_variables_names[variable]}.png')
            plt.close()

    return 0



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


    dataset_path     = dataset_config['dataset']['corrected_dataset_path']
    validation_years = dataset_config['samples']['validation_years']
    train_years      = dataset_config['samples']['train_years']
    test_years       = dataset_config['samples']['test_years']
    test_countries   = dataset_config['samples']['test_countries']
    ex_count_test    = dataset_config['samples']['exlude_countries_from_test']
    checkpoints      = train_config['model']['checkpoints']
    num_filters      = train_config['model']['num_filters']
    kernel_size      = train_config['model']['kernel_size']
    pool_size        = train_config['model']['pool_size']
    use_batchnorm    = train_config['model']['use_batchnorm']
    final_activation = train_config['model']['final_activation']
    num_layers       = train_config['model']['num_layers']
    threshold        = train_config['model']['threshold']
    drop_out_rate    = train_config['model']['drop_out_rate']
    checkpoint_path  = train_config['testing']['checkpoint_path']
    xAI_save_path     = train_config['testing']['xAI_save_path']

   

    if train_years == 'all':
        # find all available years, exclude validation and test years and use the rest for training
        all_years = os.listdir(dataset_path)
        train_years = [year for year in all_years if year not in [validation_years, test_years]]
    else:
        train_years = [train_years]
        train_years = train_years[0].split(', ')

    validation_years = [validation_years]
    validation_years = validation_years[0].split(', ')

    test_years = [test_years]
    test_years = test_years[0].split(', ')


    # get list of available countries in the dataset
    all_countries = []
    for country, value in countries_config['countries'].items():
        if value == 'True':
            all_countries.append(country)
        else:
            print(f'Country: {country} not found or disabled in available_countries.yaml')    

    # find train countries and exclude some if defined in dataset.yaml
    if test_countries == 'all':
        test_countries = all_countries
    else:
        test_countries = [test_countries]
        test_countries = test_countries[0].split(', ')

    if ex_count_test != '':
        ex_count_test = [ex_count_test]
        ex_count_test = ex_count_test[0].split(', ')
        test_countries = list(set(test_countries) - set(ex_count_test))

    
    
    test(
        dataset_path, checkpoints, 
        ast.literal_eval(num_filters), ast.literal_eval(kernel_size),
        ast.literal_eval(pool_size), bool(use_batchnorm), 
        ast.literal_eval(final_activation), float(drop_out_rate),
        train_years, validation_years, 
        float(threshold), int(num_layers),
        checkpoint_path, test_countries,
        xAI_save_path
        )
