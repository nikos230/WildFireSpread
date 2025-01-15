import torch
from torch.utils.data import DataLoader
from utils.dataset_unet3d import BurnedAreaDataset
from unet.model import UNet2D, UNet2D_struct_
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


def test(dataset_path, checkpoints, num_filters, kernel_size, pool_size, use_batchnorm, final_activation, drop_out_rate, train_years, validation_years, threshold, num_layers, checkpoint_path, test_countries):
    
    # load train and validation files from folders | Dataset and Dataloader
    test_files = load_files(dataset_path, test_years, test_countries)

    test_dataset = BurnedAreaDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Number of test samples: {len(test_loader)}')
    # Model
    input_channels = test_dataset[0][0].shape[0]
    output_channels = 1

    # model = UNet2D(
    #     in_channels=input_channels,
    #     out_channels=output_channels, 
    #     num_filters=num_filters, 
    #     kernel_size=kernel_size, 
    #     pool_size=pool_size, 
    #     use_batchnorm=use_batchnorm, 
    #     final_activation=final_activation,
    #     dropout_rate=drop_out_rate,
    #     num_layers=num_layers
    #     )

    model = UNet2D_struct_(
        in_channels=input_channels,
        out_channels=output_channels, 
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

    dice = 0
    accuracy_ = 0
    f1_score_ = 0
    iou_ = 0
    precision_ = 0
    recall_ = 0

    cnt = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)  # Shape: [batch_size, time_steps, channels, height, width]
            targets = targets.to(device).unsqueeze(1)  # Shape: [batch_size, 1, height, width]

            batch_size, channels, time_steps, height, width = inputs.shape

            time_step_losses = []
            sequence_length = inputs.size(2)

            # Iterate through each sample in the batch
            for batch_index in range(inputs.size(0)):

                # Initialize previous_output for each input sequence in the batch
                previous_output = torch.zeros(1, 1, inputs.size(3), inputs.size(4)).to(inputs.device)

                # Iterate over the time steps
                for t in range(sequence_length):
                    input_t = inputs[batch_index, :, t, :, :].unsqueeze(0)  # Select the t-th time step
                    input_t = torch.cat((input_t, previous_output), dim=1)  # Concatenate with previous output

                    # Forward pass through the model
                    output_t = model(input_t)  # Prediction for the current time step

                    # Use the prediction as the previous_output for the next time step
                    previous_output = output_t.detach()  # Detach to avoid backpropagation

                targets_ = targets[batch_index].unsqueeze(0)

                # Calculate metrics for each time step
                dice += dice_coefficient(output_t, targets_, threshold=threshold).item()
                accuracy_ += accuracy(output_t, targets_, threshold=threshold).item()
                f1_score_ += f1_score(output_t, targets_, threshold=threshold).item()
                iou_ += iou(output_t, targets_, threshold=threshold).item()
                precision_ += precision(output_t, targets_, threshold=threshold).item()
                recall_ += recall(output_t, targets_, threshold=threshold).item()
                cnt += 1
                # Apply sigmoid activation to get probabilities
                preds = torch.sigmoid(output_t)
                preds = preds.cpu().numpy()[0, 0]  # Shape: (height, width)
                targets_single = targets_[batch_index].cpu().numpy()[0, 0]  # Shape: (height, width)

                # Threshold the predictions to get binary masks
                binary_preds = (preds > threshold).astype(np.uint8)

                # Collect predictions and ground truths for visualization
                all_predictions.append(binary_preds)
                all_ground_truths.append(targets_single)
                all_dice.append(dice)


    # Calculate average metrics across all batches
    avg_dice = dice / len(test_loader) * time_steps
    avg_dice = dice / cnt
    avg_accuracy = accuracy_ / cnt
    avg_f1_score = f1_score_ / cnt
    avg_iou = iou_ / cnt
    avg_precision = precision_ / cnt
    avg_recall = recall_ / cnt

    # Print the results
    print(f'Average Dice Coefficient on Test Set: {avg_dice:.4f}')
    print(f'Average Accuracy Coefficient on Test Set: {avg_accuracy:.4f}')
    print(f'Average F1 Score on Test Set: {avg_f1_score:.4f}')
    print(f'Average IoU on Test Set: {avg_iou:.4f}')
    print(f'Average Precision on Test Set: {avg_precision:.4f}')
    print(f'Average Recall on Test Set: {avg_recall:.4f}')
    #exit()
    # Plot all predictions, ground truths, and the overlap
    num_samples = len(all_predictions)
    n_cols = 3  # 1 for prediction, 1 for ground truth, 1 for overlap
    n_rows = len(test_files)#num_samples

    fig, axes = plt.subplots(100, n_cols, figsize=(15, 5 * 100))
    
    world_boundarys = gpd.read_file('world_shapefile/world-administrative-boundaries.shp', engine='pyogrio', use_arrow=True)
    

    for i in range(0, n_rows):
        file_name = os.path.basename(test_files[i]).split('.')[0]
        # Predicted mask
        #axes[i, 0].imshow(all_predictions[i], cmap='gray', interpolation='nearest')
        #axes[i, 0].set_title(f'Sample {i+1} - Predicted Mask, File: {file_name}')

        # Ground Truth mask
        #axes[i, 1].imshow(all_ground_truths[i], cmap='gray', interpolation='nearest')
        #axes[i, 1].set_title(f'Sample {i+1} - Ground Truth Mask')

        # Overlay of prediction and ground truth with transparency
        #axes[i, 2].imshow(all_ground_truths[i], cmap='Greens', interpolation='nearest', alpha=0.5)
        #axes[i, 2].imshow(all_predictions[i], cmap='Reds', interpolation='nearest', alpha=0.5)
        #axes[i, 2].set_title(f'Sample {i+1} - Overlap (Prediction & Label)')

        transform, country, date, year, burned_area_ha = get_metadata(xr.open_dataset(test_files[i]))
        #print(burned_area_ha)
        # Save predicted mask as shapefile
        predicted_shapefile = f'out_shapefiles/predicted_shapefiles/{file_name}_predicted.shp'
        os.makedirs('out_shapefiles/predicted_shapefiles', exist_ok=True)  # Create directory if not exists
        
        mask_to_shapefile(all_predictions[i], transform, predicted_shapefile, '4326', threshold, file_name, all_dice[i], country, date, year, 0)
        
        # Save ground truth mask as shapefile
        #ground_truth_shapefile = f'out_shapefiles/groud_truth_shapefiles/{file_name}_ground_truth.shp'
        #os.makedirs('out_shapefiles/groud_truth_shapefiles', exist_ok=True)  # Create directory if not exists
        #mask_to_shapefile(all_ground_truths[i], transform, ground_truth_shapefile, '4326', threshold, file_name, all_dice[i], country, date, year, burned_area_ha)

        #for ax in axes[i]:
            #ax.axis('off')  # Hide axis

    #plt.tight_layout()

    # Save the figure with all results
    output_path = 'output_plots/test_results_with_overlap.png'
    #plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

    print(f'Saved visualization of test samples (with overlap) to {output_path}')

    #combine_shp('out_shapefiles/groud_truth_shapefiles', 'out_shapefiles/ground_truth_combined/ground_truth_combined.shp')
    combine_shp('out_shapefiles/predicted_shapefiles', 'out_shapefiles/predicted_combined/predicted_combined.shp')
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


    dataset_path     = dataset_config['dataset']['corrected_dataset_path']
    validation_years = dataset_config['samples']['validation_years']
    train_years       = dataset_config['samples']['train_years']
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

    test_years = test_years + validation_years


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


    test(dataset_path, checkpoints, ast.literal_eval(num_filters), ast.literal_eval(kernel_size), ast.literal_eval(pool_size), bool(use_batchnorm), ast.literal_eval(final_activation), float(drop_out_rate), train_years, validation_years, float(threshold), int(num_layers), checkpoint_path, test_countries)
