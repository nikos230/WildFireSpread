import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


if __name__ == "__main__":
    os.system("clear")
    
    path_to_tifs = 'output/xAI/UNet3D/sample_corrected_sample_10333_tif'
    path_to_nc = '~/nvme1/n.anastasiou/dataset_64_64_all_10days_final/2022/Greece/corrected_sample_10333.nc'

    # open the sample to make the burned area mask
    ds = xr.open_dataset(path_to_nc)
    mask = ds['burned_areas'].values[4, :, :]



    variable_sums = {}
    days_sum = {}
    ndvi_per_day_sum = {}

    positives_variables = {}
    negatives_variables = {}

    # take mean value of every saliency map (variables)
    for day in os.listdir(path_to_tifs):
        path_day = os.path.join(path_to_tifs, day)
        for variable in os.listdir(path_day):
            path_variable = os.path.join(path_day, variable)
            
            # get day and variable name
            variable_name = os.path.basename(path_variable).split('.')[0]
            day = os.path.basename(path_day)

            # open tif file 
            with rio.open(path_variable, mode='r') as variable_tif:
                variable_data = variable_tif.read(1)
                
                # add variable to dictionary if not exists
                # variable sums
                if variable_name not in variable_sums:
                    variable_sums[variable_name] = np.zeros_like(variable_data, dtype=np.float64)
                variable_sums[variable_name] += variable_data


                # days sum
                if day not in days_sum:
                    days_sum[day] = np.zeros_like(variable_data, dtype=np.float64)    
                days_sum[day] += variable_data


                if variable_name == 'saliency_map_ndvi':
                    # ndvi per day sum
                    if day not in ndvi_per_day_sum:
                        ndvi_per_day_sum[day] = np.zeros_like(variable_data, dtype=np.float64)
                    ndvi_per_day_sum[day] += variable_data

                
                # positives / negatives explain
                positives = np.where(mask, variable_data, 0)
                negatives = np.where(~mask.astype(bool), variable_data, 0)

                if variable_name not in positives_variables:
                    positives_variables[variable_name] = np.zeros_like(0, dtype=np.float64)
                positives_variables[variable_name] += positives.max()  

                if variable_name not in negatives_variables:
                    negatives_variables[variable_name] = np.zeros_like(0, dtype=np.float64)
                negatives_variables[variable_name] += negatives.max() 
  
                
                variable_tif.close()

    # for variable in positives_variables:
    #     print(positives_variables[variable])
    #     exit()



    for day in ndvi_per_day_sum:
        ndvi_per_day_sum[day] = ndvi_per_day_sum[day].sum()

    for variable in variable_sums:
        variable_sums[variable] = variable_sums[variable].sum()
        #print(np.sum(variable_sums[variable]))
        #exit()
        variable_sums[variable] = variable_sums[variable] / 10

    for day in days_sum:
        days_sum[day] = days_sum[day].sum()    

    
    # plot variables importance
    variable_sums = {key.replace('saliency_map_', ''): value for key, value in variable_sums.items()}
    positives_variables = {key.replace('saliency_map_', ''): value for key, value in positives_variables.items()}
    negatives_variables = {key.replace('saliency_map_', ''): value for key, value in negatives_variables.items()}

    del positives_variables['ignition_points']
    del negatives_variables['ignition_points']

    del variable_sums['ignition_points']

    variable_names = list(variable_sums.keys())
    variable_values = list(variable_sums.values())

    plt.figure(figsize=(12, 8))
    plt.bar(variable_names, variable_values, color='#1f77b4', zorder=0)
    plt.locator_params(axis='y', nbins=12)
    plt.title('Importance of each Variable', fontsize=14)
    plt.xlabel('Variables', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('Importance', fontsize=12)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=0)

    os.makedirs('output/xAI/plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_variables.png', dpi=300)
    plt.close()


    # plot days importance
    days_sum = {key.replace('day_', ''): value for key, value in days_sum.items()}

    days_order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    
    days_names = [day for day in days_order if day in days_sum.keys()]
    days_values = [days_sum[day] for day in days_sum]


    #plt.figure(figsize=(12, 8))
    plt.bar(days_names, days_values, color='#1f77b4', zorder=2)
    plt.locator_params(axis='y', nbins=12)
    plt.title('Importance of each Day', fontsize=10)
    plt.xlabel('Days', fontsize=9)
    plt.xticks(rotation=0)
    plt.ylabel('Importance', fontsize=9)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)

    os.makedirs('output/xAI/plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_days.png', dpi=300)
    plt.close()


    # plot ndvi per day
    days_values = [ndvi_per_day_sum[day] for day in ndvi_per_day_sum]

    plt.bar(days_names, days_values, color='#1f77b4', zorder=2)
    plt.locator_params(axis='y', nbins=12)
    plt.title('Importance of each Day', fontsize=10)
    plt.xlabel('Days', fontsize=9)
    plt.xticks(rotation=0)
    plt.ylabel('Importance', fontsize=9)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)

    os.makedirs('output/xAI/plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_ndvi_per_day.png', dpi=300)
    plt.close()


    # plot positives / negatives explain
    variable_names = list(positives_variables.keys())

    variable_values_positives = list(positives_variables.values())
    variable_values_negatives = list(negatives_variables.values())
    variable_values_negatives = -np.array(variable_values_negatives)

    plt.barh(variable_names, variable_values_negatives, color='red', label="Negatives", zorder=3)
    plt.barh(variable_names, variable_values_positives, color='blue', label="Positives", zorder=3)


    plt.axvline(0, color='black', linewidth=1)  # Add center vertical line
    plt.xlabel('Variable Importance', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.title('Importance of Each Variable (Positives vs Negatives)', fontsize=14)
    plt.legend()
    plt.grid(axis='x', linestyle=':', linewidth=0.5, color='gray', zorder=0)

    os.makedirs('output/xAI/plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_variables_positives_negatives.png', dpi=300)
    plt.close()


    