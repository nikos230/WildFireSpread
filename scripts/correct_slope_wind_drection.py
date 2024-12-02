import os
import xarray as xr
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    os.system("clear")

    datacube_path = 'shared_storage/skondylatos/med_cube/med_cube.zarr'
    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_corrected_with_countries_new'
    dataset_new_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_corrected_new'
    os.makedirs(dataset_new_path, exist_ok=True)

    ds = xr.open_zarr(datacube_path)
    
    slope = ds['slope'].load()

    for year in tqdm(os.listdir(dataset_path)):
        path_year = os.path.join(dataset_path, year)
        for country in os.listdir(path_year):
            path_country = os.path.join(path_year, country)
            for sample in os.listdir(path_country):
                path_to_sample = os.path.join(path_country, sample)
                sample_name = os.path.basename(path_to_sample)

                ds_sample = xr.open_dataset(path_to_sample)

                # fix the slope
                updated_slope = ds['slope'].sel(x=ds_sample['x'], y=ds_sample['y'], method='nearest')
                ds_sample['slope'] = updated_slope
                if np.isnan(ds_sample['slope']).any():
                    ds_sample['slope'] = ds_sample['slope'].fillna(0)

                # fix the wind direction
                wind_direction = ds_sample['wind_direction']
                wind_direction_sin = np.sin(np.deg2rad(wind_direction))
                wind_direction_cos = np.cos(np.deg2rad(wind_direction))

                ds_sample['wind_direction_sin'] = wind_direction_sin
                ds_sample['wind_direction_cos'] = wind_direction_cos

                # save the new sample to new path
                path_to_new = os.path.join(os.path.join(dataset_new_path, year), country)
                os.makedirs(path_to_new, exist_ok=True)
                ds_sample.to_netcdf(path_to_new + '/' + sample_name)
                ds_sample.close()
    print('Done!')
    
