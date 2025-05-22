import torch
from torch.utils.data import DataLoader
from utils.dataset_unet2d import BurnedAreaDataset
from unet.model import UNet2D, UNet2D_struct, ViTSegmentation2
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
import shutil
#from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex


def test(
        dataset_path, checkpoints, 
        num_filters, kernel_size, 
        pool_size, use_batchnorm, 
        final_activation, drop_out_rate,
        train_years, validation_years,
        threshold, num_layers,
        checkpoint_path, test_countries,
        save_results_path, world_bounds
        ):
    
    # load train and validation files from folders | Dataset and Dataloader
    test_files = load_files(dataset_path, test_years, test_countries)

    test_dataset = BurnedAreaDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Number of test samples: {len(test_loader)}')
    # Model
    input_channels = test_dataset[0][0].shape[0]
    output_channels = 1

    model = UNet2D(
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

    # model = UNet2D_struct(
    #     in_channels=input_channels,
    #     out_channels=output_channels, 
    #     )    


    # in_channels = test_dataset[0][0].shape[0]         # 27-channel input
    # num_classes = 1          # Binary segmentation
    # image_size = 64          # Image resolution
    # patch_size = 2           # Patch size
    # embed_dim = 128     # Adjust embedding dimension for smaller input
    # num_heads = 16            # Number of attention heads
    # depth = 1             # Transformer depth
    # mlp_dim = 256              # Feedforward dimension
    # dropout_rate = 0.4

    
    # model = ViTSegmentation2(
    #     in_channels=in_channels,
    #     num_classes=num_classes,
    #     image_size=image_size,
    #     patch_size=patch_size,
    #     embed_dim=embed_dim,
    #     num_heads=num_heads,
    #     depth=depth,
    #     mlp_dim=mlp_dim,
    #     dropout_rate=dropout_rate
    # )
    
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

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    all_predictions = []
    all_ground_truths = []
    all_dice = []

    # torch metrics stuff
    # metric_kwargs = {"threshold": threshold}

    # accuracy_metric = BinaryAccuracy(**metric_kwargs)
    # precision_metric = BinaryPrecision(**metric_kwargs)
    # recall_metric = BinaryRecall(**metric_kwargs)
    # f1_metric = BinaryF1Score(**metric_kwargs)
    # iou_metric = BinaryJaccardIndex(**metric_kwargs)
    # #dice_metric = Dice(**metric_kwargs)

    # # Move to device
    # metrics = [accuracy_metric, precision_metric, recall_metric, f1_metric, iou_metric]
    # for m in metrics:
    #     m.to(device)


    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.unsqueeze(1)  # Add channel dimension

            outputs = model(inputs)

            # update torch metrics
            # for m in metrics:
            #     m.update(outputs.squeeze(0).squeeze(0), targets.squeeze(0).squeeze(0))

            dice = dice_coefficient(outputs, targets, threshold=threshold).item()
            accuracy_ = accuracy(outputs, targets, threshold=threshold).item()
            f1_score_ = f1_score(outputs, targets, threshold=threshold).item()
            iou_ = iou(outputs, targets, threshold=threshold).item()
            precision_ = precision(outputs, targets, threshold=threshold).item()
            recall_ = recall(outputs, targets, threshold=threshold).item()

            total_dice += dice
            total_accuracy += accuracy_
            total_f1_score += f1_score_
            total_iou += iou_
            total_precision += precision_
            total_recall += recall_


            # Apply sigmoid activation to get probabilities
            preds = torch.sigmoid(outputs)
            preds = preds.cpu().numpy()[0, 0]  # Shape: (height, width)
            targets = targets.cpu().numpy()[0, 0]  # Shape: (height, width)

            # Optional: Threshold the predictions to get binary masks
            binary_preds = (preds > threshold).astype(np.uint8)

            # Collect predictions and ground truths for visualization
            all_predictions.append(binary_preds)
            all_ground_truths.append(targets)
            all_dice.append(dice)


            # calcuate metrics for all test set
            
            # gather all tp, fp, tn, fn for each sample
            tp = np.logical_and(binary_preds == 1, targets == 1).sum()
            fp = np.logical_and(binary_preds == 1, targets == 0).sum()
            tn = np.logical_and(binary_preds == 0, targets == 0).sum()
            fn = np.logical_and(binary_preds == 0, targets == 1).sum()

            # add to total
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn





    # arxikos tropos
    avg_dice = total_dice / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    avg_f1_score = total_f1_score / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    avg_precision = total_precision / len(test_loader)
    avg_recall = total_recall / len(test_loader)  

    # print(f'Average Dice Coefficient on Test Set: {avg_dice:.4f}')
    # print(f'Average Accuracy Coefficient on Test Set: {avg_accuracy:.4f}')
    # print(f'Average f1 Score Coefficient on Test Set: {avg_f1_score:.4f}')
    # print(f'Average IoU Coefficient on Test Set: {avg_iou:.4f}')
    # print(f'Average Precision Coefficient on Test Set: {avg_precision:.4f}')
    # print(f'Average Recall Coefficient on Test Set: {avg_recall:.4f}')

    # calculate ver2
    eps = 1e-7  # To prevent division by zero

    avg_dice_2 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    avg_accuracy_2 = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)
    avg_precision_2 = total_tp / (total_tp + total_fp + eps)
    avg_recall_2 = total_tp / (total_tp + total_fn + eps)
    avg_f1_score_2 = (2 * (avg_precision_2 * avg_recall_2)) / (avg_precision_2 + avg_recall_2 + eps)
    avg_iou_2 = total_tp / (total_tp + total_fp + total_fn + eps)  
    print('\n')
    print(f'Average Dice Coefficient on Test Set: {avg_dice_2:.4f}')
    print(f'Average Accuracy Coefficient on Test Set: {avg_accuracy_2:.4f}')
    print(f'Average f1 Score Coefficient on Test Set: {avg_f1_score_2:.4f}')
    print(f'Average IoU Coefficient on Test Set: {avg_iou_2:.4f}')
    print(f'Average Precision Coefficient on Test Set: {avg_precision_2:.4f}')
    print(f'Average Recall Coefficient on Test Set: {avg_recall_2:.4f}')


    # torch metrics
    # avg_accuracy_3 = accuracy_metric.compute().item()
    # avg_precision_3 = precision_metric.compute().item()
    # avg_recall_3 = recall_metric.compute().item()
    # avg_f1_3 = f1_metric.compute().item()
    # avg_iou_3 = iou_metric.compute().item()
    # #avg_dice_3 = dice_metric.compute().item()

    # print('\n')
    # #print(f'Average Dice Coefficient on Test Set: {avg_accuracy_3:.4f}')
    # print(f'Average Accuracy on Test Set: {avg_accuracy_3:.4f}')
    # print(f'Average F1 Score on Test Set: {avg_f1_3:.4f}')
    # print(f'Average IoU on Test Set: {avg_iou_3:.4f}')
    # print(f'Average Precision on Test Set: {avg_precision_3:.4f}')
    # print(f'Average Recall on Test Set: {avg_recall_3:.4f}')


    #exit()

    # plot all predictions, ground truths, and the overlap
    num_samples = len(all_predictions)
    n_cols = 3  # 1 for prediction, 1 for ground truth, 1 for overlap
    os.makedirs(save_results_path, exist_ok=True)
    
    world_boundarys = gpd.read_file(world_bounds, engine='pyogrio', use_arrow=True)

    fig, axes = plt.subplots(100, n_cols, figsize=(15, 5 * 100))

    for i in range(0, 100):
        file_name = os.path.basename(test_files[i]).split('.')[0]
        # Predicted mask
        axes[i, 0].imshow(all_predictions[i], cmap='gray', interpolation='nearest')
        axes[i, 0].set_title(f'Sample {i+1} - Predicted Mask, File: {file_name}')

        # Ground Truth mask
        axes[i, 1].imshow(all_ground_truths[i], cmap='gray', interpolation='nearest')
        axes[i, 1].set_title(f'Sample {i+1} - Ground Truth Mask')

        # Overlay of prediction and ground truth with transparency
        axes[i, 2].imshow(all_ground_truths[i], cmap='Greens', interpolation='nearest', alpha=0.5)
        axes[i, 2].imshow(all_predictions[i], cmap='Reds', interpolation='nearest', alpha=0.5)
        axes[i, 2].set_title(f'Sample {i+1} - Overlap (Prediction & Label)')

    for ax in axes[i]:
        ax.axis('off')
    
    os.makedirs(save_results_path, exist_ok=True)
    output_path = f'{save_results_path}/UNet2D_test_results.png'
    plt.savefig(output_path)
    plt.close()        

    print(f'Saved visualization of test samples (with overlap) to {output_path}')



    for i in range(0, num_samples):
        file_name = os.path.basename(test_files[i]).split('.')[0]
        transform, country, date, year, burned_area_ha = get_metadata(xr.open_dataset(test_files[i]))
        
        # save predicted mask as shapefile
        predicted_shapefile = f'{save_results_path}/predicted_all_shapefiles/{file_name}_predicted.shp'
        os.makedirs(f'{save_results_path}/predicted_all_shapefiles', exist_ok=True)  
        
        mask_to_shapefile(all_predictions[i], transform, predicted_shapefile, '4326', threshold, file_name, all_dice[i], country, date, year, 0)
        
        # save ground truth mask as shapefile
        ground_truth_shapefile = f'{save_results_path}/groud_truth_all_shapefiles/{file_name}_ground_truth.shp'
        os.makedirs(f'{save_results_path}/groud_truth_all_shapefiles', exist_ok=True)  
        mask_to_shapefile(all_ground_truths[i], transform, ground_truth_shapefile, '4326', threshold, file_name, all_dice[i], country, date, year, burned_area_ha)



    combine_shp(f'{save_results_path}/groud_truth_all_shapefiles', f'{save_results_path}/groud_truth_shapefiles/UNet2D_test_ground_truth.shp')
    combine_shp(f'{save_results_path}/predicted_all_shapefiles'  , f'{save_results_path}/predicted_shapefiles/UNet2D_test_predicted.shp')

    shutil.rmtree(f'{save_results_path}/groud_truth_all_shapefiles/')
    shutil.rmtree(f'{save_results_path}/predicted_all_shapefiles/')

    print('Saved Shapefiles!')



def get_metadata(ds):
    # get transform for the out shapefile
    x_coords = ds['x'].values
    y_coords = ds['y'].values

    pixel_size_x = x_coords[1] - x_coords[0]
    pixel_size_y = y_coords[1] - y_coords[0]  

    origin_x = x_coords.min()  
    origin_y = y_coords.max()  

    transform = Affine.translation(origin_x, origin_y) * Affine.scale(pixel_size_x, pixel_size_y)

    return transform, ds.attrs['country'], ds.attrs['date'], ds.attrs['year'], ds.attrs['burned_area_ha']



def mask_to_shapefile(mask, transform, out_shapefile, epsg, threshold, sample, dice, country, date, year, burned_area_ha):

    binary_mask = (mask > threshold).astype(np.uint8)
    shapes_generator = shapes(binary_mask, mask=binary_mask, transform=transform)

    shapes_list = list(shapes_generator)
    if burned_area_ha == 0:
        #print(shapes_list)
        for i, (geom, value) in enumerate(shapes_list):
           #print('test')
            polygon = shape(geom)
            km_per_degree = 111
            burned_area_new = polygon.area * (km_per_degree**2) * 100 # ha
            burned_area_ha = burned_area_ha + burned_area_new
    
    
    data = []
    for i, (geom, value) in enumerate(shapes_list):
        if value == 1:
            data.append({
                'geometry': shape(geom),
                'value': int(value),
                'sample': str(sample),
                'dice': float(dice),
                'date': str(date),
                'year': str(year),
                'country': str(country),
                'b_area_ha': float(burned_area_ha)
            })
    if not data:
        return 0
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=epsg, inplace=True)
    gdf_dissloved = gdf.dissolve('sample')
    gdf_dissloved.to_file(out_shapefile, driver='ESRI Shapefile')



def combine_shp(input_folder, out_shapefile):
    os.makedirs(os.path.dirname(out_shapefile), exist_ok=True)
    gdfs = []
    for shapefile in os.listdir(input_folder):
        if shapefile.endswith('.shp'):
            file_path = os.path.join(input_folder, shapefile)
            gdf = gpd.read_file(file_path)
            gdfs.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    combined_gdf.to_file(out_shapefile, driver='ESRI Shapefile')        



if __name__ == '__main__':
    os.system("clear")

    with open('configs/train_test_config_unet2d.yaml', 'r') as t_config:
        train_config = yaml.safe_load(t_config)
    t_config.close()

    with open('configs/dataset.yaml', 'r') as d_config:
        dataset_config = yaml.safe_load(d_config)
    d_config.close()    

    with open('configs/available_countries.yaml', 'r') as c_config:
        countries_config = yaml.safe_load(c_config)
    c_config.close()


    dataset_path      = dataset_config['dataset']['corrected_dataset_path']
    world_bounds      = dataset_config['dataset']['world_countries_bounds']
    validation_years  = dataset_config['samples']['validation_years']
    train_years       = dataset_config['samples']['train_years']
    test_years        = dataset_config['samples']['test_years']
    test_countries    = dataset_config['samples']['test_countries']
    ex_count_test     = dataset_config['samples']['exlude_countries_from_test']
    checkpoints       = train_config['model']['checkpoints']
    num_filters       = train_config['model']['num_filters']
    kernel_size       = train_config['model']['kernel_size']
    pool_size         = train_config['model']['pool_size']
    use_batchnorm     = train_config['model']['use_batchnorm']
    final_activation  = train_config['model']['final_activation']
    num_layers        = train_config['model']['num_layers']
    threshold         = train_config['model']['threshold']
    drop_out_rate     = train_config['model']['drop_out_rate']
    checkpoint_path   = train_config['testing']['checkpoint_path']
    save_results_path = train_config['testing']['save_results_path']

   

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
        save_results_path, world_bounds
        )
