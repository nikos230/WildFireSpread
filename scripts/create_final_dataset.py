import os
import xarray as xr
import numpy as np

def check_nc_file_consistency_and_fix(root_folder_path, corrected_root_folder_path):
    # List of years (subfolders)
    year_folders = [f for f in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, f))]

    for year_folder in year_folders:
        folder_path = os.path.join(root_folder_path, year_folder)
        corrected_folder_path = corrected_root_folder_path
        
        # Create the corrected directory for the year if it doesn't exist
        os.makedirs(corrected_folder_path, exist_ok=True)

        # List all .nc files in the folder
        nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
        nc_files.sort()

        if not nc_files:
            print(f"No .nc files found in the folder for year {year_folder}.")
            continue
        
        print(f'Total number of files in: {folder_path} for year {year_folder} is {len(nc_files)}')

        # Initialize reference variables and dimensions
        reference_data_vars = None
        reference_dims = {'x': 64, 'y': 64, 'time': 4}
        final_files = 0

        for i, nc_file in enumerate(nc_files):
            file_path = os.path.join(folder_path, nc_file)

            # Open the dataset using xarray
            try:
                ds = xr.open_dataset(file_path)
            except Exception as e:
                print(f"Error opening {nc_file}: {e}")
                continue
            
            # Extract the data variables and dimensions
            current_data_vars = list(ds.data_vars)
            current_dims = {dim: ds.dims[dim] for dim in ['y', 'x', 'time'] if dim in ds.dims}

            # If this is the first file, set it as the reference
            if i == 0:
                reference_data_vars = current_data_vars
                reference_dims = current_dims
                print(f"Reference variables and dimensions set from {nc_file}")
            else:
                # Check if the current file has the same variables
                if current_data_vars != reference_data_vars:
                    print(f"Data variables mismatch in file: {nc_file} for year {year_folder}")
                    print(f"Expected: {reference_data_vars}, but got: {current_data_vars}")
                    print(f"Skipping file: {nc_file}")
                    continue  # Skip this file if data variables mismatch
                
                # Check if the current file has the same dimensions
                if current_dims != reference_dims:
                    print(f"Dimension mismatch in file: {nc_file} for year {year_folder}")
                    print(f"Expected: {reference_dims}, but got: {current_dims}")
                    print(f"Skipping file: {nc_file}")
                    continue  # Skip this file if dimensions mismatch
            
            # Check for NaN values in the dataset
            skip_file = False

            for variable in ds.data_vars:
                data = ds[variable]

                # Check if all values are NaN
                if np.isnan(data).all():
                    print(f"All values in {variable} in file {nc_file} are NaN, skipping file...")
                    skip_file = True
                    break
                
                # Check for partial NaN values and fix them
                if np.isnan(data).any():
                    print(f"Found NaN values in {variable} in file {nc_file}, fixing...")
                    ds[variable] = data.fillna(0)  # Fill NaN values with 0
            
            if skip_file:
                print(f'Skipped file: {nc_file} for year {year_folder}')
                continue

            # Save the corrected dataset to the new folder
            corrected_file_path = os.path.join(corrected_folder_path, f"corrected_{nc_file}")
            ds.to_netcdf(corrected_file_path)
            final_files += 1
            print(f'Processed and saved file: {nc_file} for year {year_folder}')

        print(f"Total number of files processed and saved for year {year_folder}: {final_files}")


# Example usage
root_folder_path = "WildFireSpread/dataset_mesogeos"
corrected_root_folder_path = "WildFireSpread/dataset_mesogeos_corrected"
check_nc_file_consistency_and_fix(root_folder_path, corrected_root_folder_path)
