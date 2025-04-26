import os
import xarray as xr
import numpy as np
from shapely.geometry import shape, Point
from rasterio.features import shapes
import geopandas as gpd
from affine import Affine
import pandas as pd
import shutil


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



def mask_to_shapefile_as_points(mask, transform, out_shapefile, epsg, threshold, sample, dice, country, date, year, burned_area_ha):
    # Convert mask to binary
    #mask = mask.values
    binary_mask = (mask >= 1).astype(np.uint8)
    shapes_generator = shapes(binary_mask, mask=binary_mask, transform=transform)

    # Initialize burned area calculation
    shapes_list = list(shapes_generator)


    data = []
    for i, (geom, value) in enumerate(shapes_list):
        if value == 1:
            polygon = shape(geom)
            centroid = polygon.centroid  # Convert polygon to its centroid point
            data.append({
                'geometry': centroid,  # Use the centroid as the geometry
                'value': int(value),
                'sample': str(sample),
                'dice': float(dice),
                'date': str(date),
                'year': str(year),
                'country': str(country),
                'b_area_ha': float(burned_area_ha)
            })

    if not data:
        print('no data found')
        return 0

    # Create GeoDataFrame from point data
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=epsg, inplace=True)
    gdf.to_file(out_shapefile, driver='ESRI Shapefile')


def combine_shp(input_folder, out_shapefile):
    os.makedirs(os.path.dirname(out_shapefile), exist_ok=True)
    gdfs = []
    for folder in os.listdir(input_folder):
        for shapefile in os.listdir(os.path.join(input_folder, folder)):
            if shapefile.endswith('.shp'):
                file_path = os.path.join(input_folder, os.path.join(folder, shapefile))

                gdf = gpd.read_file(file_path)
                gdfs.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    combined_gdf.to_file(out_shapefile, driver='ESRI Shapefile')  
    print(f'Combined Shapefile Saved')


if __name__ == '__main__':
    os.system("clear")

    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final'
    output_path = 'output/Ignition_Points'

    os.makedirs(output_path, exist_ok=True)

    years = ['2022']
    
    #combine_shp('WildFireSpread/WildFireSpread_UNet2D/Ignition_Points', 'WildFireSpread/WildFireSpread_UNet2D/output_ignition_points/ignition_points.shp')
    #exit()
    for year in years:
        dataset_path_year = os.path.join(dataset_path, year)
        for country in os.listdir(dataset_path_year):
            dataset_path_year_country = os.path.join(dataset_path_year, country)
            for sample in os.listdir(dataset_path_year_country):

                ds = xr.open_dataset(os.path.join(dataset_path_year_country, sample))

                transform, country, date, year, burned_area_ha = get_metadata(ds)

                ignition_point = ds['ignition_points'].values[4]
                file_name = sample.split('.')[0]
                
                mask_to_shapefile_as_points(ignition_point, transform, f'{output_path}/Ignition_Point_{file_name}', '4326', 0.5, sample, -1, country, date, year, 0)
                

    combine_shp(output_path, f'output/test_ignition_points/test_ignition_points.shp')    

    shutil.rmtree(output_path)