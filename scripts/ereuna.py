import os
import xarray as xr
import numpy as np
import geopandas as gpd


if __name__ == "__main__":
    os.system("clear")
    # dataset path
    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_corrected_with_countries_new'
    # load shapefiles
    predicted = gpd.read_file('WildFireSpread/WildFireSpread_UNet3D/out_shapefiles/predicted_combined/predicted_combined.shp')
    label = gpd.read_file('WildFireSpread/WildFireSpread_UNet3D/out_shapefiles/ground_truth_combined/ground_truth_combined.shp')

    # sort rows by dice column high to low
    label = label.sort_values(by=['dice'], ascending=False)
    predicted = predicted.sort_values(by=['dice'], ascending=False)
    
    # katigories < 40, 40-60%, 60-80%, 80-100% 
    predicted_0_40 = predicted[ ( predicted['dice'] < 0.40 ) ]
    predicted_40_60 = predicted[ ( predicted['dice'] > 0.40 ) & ( predicted['dice'] < 0.60 ) ]
    predicted_60_80 = predicted[ ( predicted['dice'] > 0.60 ) & ( predicted['dice'] < 0.80 ) ]
    predicted_80_100 = predicted[ ( predicted['dice'] > 0.80 ) ]

    output_path = 'WildFireSpread/WildFireSpread_UNet3D/out_shapefiles/ereuna'
    os.makedirs(output_path, exist_ok=True)


    num_of_samples = 0

    total_burned_area = 0
    min_burned_area = 1000000
    max_burned_area = 0

    total_wind_speed = 0
    total_ndvi = 0
    total_wind_direction = 0
    total_dem = 0
    total_lai = 0
    total_lst_day = 0
    total_rh = 0
    total_t2m = 0
    total_d2m = 0

    for i in range(len(predicted_0_40)):

        year = predicted_0_40.iloc[i]['year']
        sample = predicted_0_40.iloc[i]['sample']
        country = predicted_0_40.iloc[i]['country']

        path_to_sample = os.path.join(os.path.join(os.path.join(dataset_path, year), country), sample+'.nc')
        ds = xr.open_dataset(path_to_sample)

        num_of_samples += 1
        burned = ds.attrs['burned_area_ha']
        if min_burned_area >  burned:
            min_burned_area = burned
        if max_burned_area < burned:
            max_burned_area = burned

        total_burned_area += ds.attrs['burned_area_ha']
        #print(ds.attrs['burned_area_ha'])
        total_wind_speed += ds['wind_speed'].values[4].mean()
        total_ndvi += ds['ndvi'].values[4].mean()
        total_wind_direction += ds['wind_direction'].values[4].mean()
        total_dem += ds['dem'].values[4].mean()
        total_lai += ds['lai'].values[4].mean()
        total_lst_day += ds['wind_direction'].values[4].mean()
        total_rh += ds['rh'].values[4].mean()
        total_t2m += ds['t2m'].values[4].mean()
        total_d2m += ds['d2m'].values[4].mean()

        print(f'burned area:{burned}, country:{country}, year:{year}, sample:{sample}')
    print(f'Mean Burned Area: {total_burned_area / num_of_samples}\nMin Burned Area: {min_burned_area}\nMax Burned Area: {max_burned_area}\nMean Wind Speed: {total_wind_speed/num_of_samples}')
    print(f'Mean ndvi: {total_ndvi/num_of_samples}\nMean Total Wind Direction: {total_wind_direction/num_of_samples}\nMean Dem:{total_dem/num_of_samples}\nMean Lai:{total_lai/num_of_samples}')
    print(f'Mean lst day:{total_lst_day/num_of_samples}\nMean rh:{total_rh/num_of_samples}\nMean t2m:{total_t2m/num_of_samples}\nMean d2m:{total_d2m/num_of_samples}')  
    print(num_of_samples)  

     

    #predicted_0_40.to_file(output_path + '/' + 'predicted_0_40.shp', driver='ESRI Shapefile')
    #predicted_40_60.to_file(output_path + '/' + 'predicted_40_60.shp', driver='ESRI Shapefile')
    #predicted_60_80.to_file(output_path + '/' + 'predicted_60_80.shp', driver='ESRI Shapefile')
    #predicted_80_100.to_file(output_path + '/' + 'predicted_80_100.shp', driver='ESRI Shapefile')
    
    
