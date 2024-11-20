import os
import xarray as xr
import numpy as np
from tqdm import tqdm
import yaml
import ast
import geopandas as gpd
from shapely.geometry import Point


def check_dims_and_vars(ds, reference_dims):
    reference_data_vars = None
    reference_dims = reference_dims
    # get currect dims for the nc file
    current_data_vars = list(ds.data_vars)
    current_dims = ds.dims#{dim: ds.dims[dim] for dim in ['y', 'x', 'time'] if dim is ds.dims}
    
    # check if they are correct, if not skip sample
    if current_dims != reference_dims: #or current_data_vars != reference_data_vars:
        return False
    else:
        return True    



def check_for_nans(ds, skip=False):
    # get all variables from the sample and check for nans and correct them
    for variable in ds.data_vars:
        data = ds[variable]

        # check if all variable is nan, then skip it
        if np.isnan(data).all():
            skip = True
            break

        # check if any value if nan and replace it with 0
        if np.isnan(data).any():
            ds[variable] = data.fillna(0)

    return ds, skip            


def get_metadata(ds, world_bound):

    # get country name
    x_center = ds['x'].values.mean()
    y_center = ds['y'].values.mean()

    center_point = Point(x_center, y_center)

    country = world_bound[world_bound.contains(center_point)]

    if not country.empty:
        country_name = country.iloc[0]['name']
        #country_name = country_name.replace(" ", "_").lower()
    else: # pernei tin pio kontini xwra allios
        world_bound['distance'] = world_bound.geometry.apply(lambda geom: geom.distance(center_point))
        nearest_country = world_bound.loc[world_bound['distance'].idxmin()]
        country_name = nearest_country['name']   


    # get fire day date and year
    times = ds['time'].values
    date = np.datetime64(times[4]) # 5 days before and 2 after (with fire day inside)

    year = date.astype('datetime64[Y]').item().year
    date = str(date).split('T')[0]     

    # find burned area size in ha
    x_res = abs(ds['x'][1] - ds['x'][0])
    y_res = abs(ds['y'][1] - ds['y'][0])
    pixel_area = x_res * y_res
    
    #if ds['burned_areas'].dtype == np.bool_ or np.all(np.isin(ds['burned_areas'], [0, 1])):
        #burned_pixels = ds['burned_areas'].values[4].sum()
        
        #total_burned_area = burned_pixels * pixel_area
   # else:
    total_burned_area = ds['burned_areas'].values[4].sum() * pixel_area

    km_per_degree = 111 # auto einai sxedon lathos na to tsekarw kapoia mera
    total_burned_area = total_burned_area * (km_per_degree**2) * 100 # se ha

    if total_burned_area > 100000000:
        print('opa re')
        exit()


    return country_name, date, year, round(float(total_burned_area), 0)



def create_final_dataset(root_dataset, corrcted_dataset, reference_dims, world_bound):
    corrcted_files = 0
    skipped_files = 0
    # create output folder if not exist
    os.makedirs(corrcted_dataset, exist_ok=True)

    # get sample names by year
    
    # first get names of year folders
    years_folders = [f for f in os.listdir(root_dataset) if os.path.isdir(os.path.join(root_dataset, f))]

    for year_folder in tqdm(years_folders):
        # create output folder for every year for the corrcted dataset
        os.makedirs(os.path.join(corrcted_dataset, year_folder), exist_ok=True)

        # get all .nc files (samples) from the original folder
        nc_files = [f for f in os.listdir(os.path.join(root_dataset, year_folder)) if f.endswith('.nc')]
        nc_files.sort()

        if not nc_files:
            print(f'No Files are Found! in {year_folder}')
            continue

        
        # get all nc files from the currect year folder
        for i, nc_file in enumerate(nc_files):
            file_path = os.path.join(os.path.join(root_dataset, year_folder), nc_file)

            ds = xr.open_dataset(file_path)

            if check_dims_and_vars(ds, reference_dims) == False:
                skipped_files += 1
                continue # skip tje current sample if it has damaged dims or data variables
            
            ds, skip = check_for_nans(ds)
            if skip == True:
                skipped_files += 1
                continue # skip file because a hole variable is nan
            
            # add metadata
            country, date, year, total_burned_area = get_metadata(ds, world_bound)

            ds.attrs['country'] = country
            ds.attrs['year'] = year
            ds.attrs['date'] = date
            ds.attrs['burned_area_ha'] = total_burned_area
            
            # save corrected file
            corrcted_file_path_year = os.path.join(os.path.join(corrcted_dataset, year_folder))
            os.makedirs(corrcted_file_path_year, exist_ok=True)
            corrcted_file_path_year_country = os.path.join(corrcted_file_path_year, country)
            final_path = os.path.join(corrcted_file_path_year_country, f'corrected_{nc_file}')
            os.makedirs(corrcted_file_path_year_country, exist_ok=True)

            ds.to_netcdf(final_path)
            ds.close()
            del ds
            corrcted_files += 1

    print(f'Done!, total samples: {corrcted_files}\n skipped samples: {skipped_files}')        
            


if __name__ == "__main__":
    os.system('clear')

    with open('WildFireSpread/WildFireSpread_UNet3D/configs/dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    file.close()

    # root dataset as been created from dataset_extraction.py and corrected dataset path to be ssaved on (change in configs/dataset.yaml)
    root_dataset     = config['dataset']['dataset_path']
    corrcted_dataset = config['dataset']['corrected_dataset_path']
    reference_dims   = config['dataset']['reference_dims']
    world_bounds_shp = config['dataset']['world_countries_bounds']   

    # load shapefile with world boundaries
    world_bound = gpd.read_file(world_bounds_shp, engine='pyogrio', use_arrow=True)

    create_final_dataset(root_dataset, corrcted_dataset, ast.literal_eval(reference_dims), world_bound)