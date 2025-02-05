import os
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np


if __name__ == "__main__":
    os.system("clear")

    dataset_path = '~/nvme1/n.anastasiou/dataset_64_64_all_10days_final'
    
    predicted = gpd.read_file('output/output_UNet3D/predicted_combined/predicted_combined.shp')
    predicted = predicted.sort_values(by=['dice'], ascending=False)

    print(predicted.columns)
    
    # plot burned area and dice in scatter plot
    # plt.scatter(predicted['dice'], predicted['b_area_ha'], color='b', s=10)
    # plt.xlabel('dice')
    # plt.ylabel('burned area')

    # plt.savefig('output/test.png', dpi=300)

    # plot ndvi and dice in scatter plot
    #ndvi_dice_data = pd.DataFrame(columns=['dice', 'ndvi'])
    ndvi = []
    dice = []

    for sample in range(0, len(predicted)):
        country = predicted.iloc[sample]['country']
        year = predicted.iloc[sample]['year']
        sample_name = predicted.iloc[sample]['sample'] + '.nc'

        path = os.path.join(dataset_path, year)
        path = os.path.join(path, country)
        path = os.path.join(path, sample_name)
        
        ds = xr.open_dataset(path)

        ndvi_ = ds['ndvi'].values[4, :, :]
        lst_day_ = ds['lst_day'].values[4, :, :]
        
        burned_areas = ds['burned_areas'].values[4, :, :]
        burned_areas_ = burned_areas > 0

        lst_day_ = np.mean(lst_day_[burned_areas_])
        ndvi_ = np.mean(ndvi_[burned_areas_])

        if lst_day_ == 0:
            continue
        ndvi_ = ndvi_ + lst_day_
        #ndvi_dice_data.iloc[sample]['ndvi'] = ndvi
        #ndvi_dice_data.iloc[sample]['dice'] = predicted.iloc[sample]['dice']
        ndvi.append(ndvi_)
        dice.append(predicted.iloc[sample]['dice'])
        #print(predicted.iloc[sample]['dice'])

    plt.scatter(dice, ndvi, color='b', s=10)
    plt.xlabel('dice')
    plt.ylabel('ndvi')

    plt.savefig('output/test_2.png', dpi=300)    


        