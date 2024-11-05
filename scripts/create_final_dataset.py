import os
import xarray as xr
import numpy as np
from tqdm import tqdm
import yaml


def check_dims_and_vars(ds):
    reference_data_vars = None
    reference_dims = {'y': 64, 'x': 64, 'time': 6}
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



def create_final_dataset(root_dataset, corrcted_dataset):
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

            if check_dims_and_vars(ds) == False:
                skipped_files += 1
                continue # skip tje current sample if it has damaged dims or data variables
            
            ds, skip = check_for_nans(ds)
            if skip == True:
                skipped_files += 1
                continue # skip file because a hole variable is nan
            
            # save corrected file
            corrcted_file_path = os.path.join(os.path.join(corrcted_dataset, year_folder), f'corrected_{nc_file}')
            ds.to_netcdf(corrcted_file_path)
            ds.close()
            del ds
            corrcted_files += 1

    print(f'Done!, total samples: {corrcted_files}\n skipped samples: {skipped_files}')        
            


if __name__ == "__main__":

    with open('WildFireSpread/WildFireSpread_UNET/configs/dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    file.close()

    # root dataset as been created from dataset_extraction.py and corrected dataset path to be ssaved on (change in configs/dataset.yaml)
    root_dataset     = config['dataset']['dataset_path']
    corrcted_dataset = config['dataset']['corrected_dataset_path']     

    create_final_dataset(root_dataset, corrcted_dataset)