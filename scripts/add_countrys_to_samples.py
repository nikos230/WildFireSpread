import xarray as xr
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def get_metadata(ds, world_bound):

    # get country name
    x_center = ds['x'].values.mean()
    y_center = ds['y'].values.mean()

    center_point = Point(x_center, y_center)

    country = world_bound[world_bound.contains(center_point)]

    if not country.empty:
        country_name = country.iloc[0]['name']
    else:
        country_name = 'null'   


    # get fire day date and year
    times = ds['time'].values
    date = np.datetime64(times[4]) # 5 days before and 2 after (with fire day inside)

    year = date.astype('datetime64[Y]').item().year
    date = str(date).split('T')[0]     

    # find burned area size in ha
    x_res = abs(ds['x'][1] - ds['x'][0])
    y_res = abs(ds['y'][1] - ds['y'][0])
    pixel_area = x_res * y_res
    
    if ds['burned_areas'].dtype == np.bool_ or np.all(np.isin(ds['burned_areas'], [0, 1])):
        burned_pixels = ds['burned_areas'].sum().item()
        total_burned_area = burned_pixels * pixel_area
    else:
        total_burned_area = ds['burned_areas'].sum().item()

    km_per_degree = 111 # auto einai sxedon lathos na to tsekarw kapoia mera
    total_burned_area = total_burned_area * (km_per_degree**2) * 100 # se ha


    return country_name, date, year, round(float(total_burned_area), 0)


if __name__ == '__main__':
    os.system('clear')

    # load world boundarys shapefile
    world_bound = gpd.read_file('hdd1/n.anastasiou/WildFireSpread/WildFireSpread_UNet3D/world_shapefile/world-administrative-boundaries.shp', engine='pyogrio', use_arrow=True)

    dataset_path = 'nvme1/n.anastasiou/dataset_64_64_all_corrected'

    for year in os.listdir(dataset_path):
        path_to_samples = os.path.join(dataset_path, year)
        for sample in os.listdir(path_to_samples):
            path_to_sample = os.path.join(path_to_samples, sample)
            ds = xr.open_dataset(path_to_sample)
            country, date, year, total_burned_area = get_metadata(ds, world_bound)
            ds.attrs['country'] = country
            ds.attrs['year'] = year
            ds.attrs['date'] = date
            ds.attrs['burned_area_ha'] = total_burned_area
            print(ds.attrs)
            exit()
            ds.to_netcdf(path_to_sample)