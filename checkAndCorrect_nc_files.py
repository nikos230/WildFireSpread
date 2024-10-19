import os
import xarray as xr
import numpy as np

# Directory where your .nc files are stored
directory = "WildFireSpread/dataset_mesogeos/2021"
corrected_directory = "WildFireSpread/dataset_mesogeos_corrected/2021"
os.makedirs(corrected_directory, exist_ok=True)

files = [file for file in os.listdir(directory) if file.endswith('.nc')]
files.sort()

print(f'Total number of files in: {directory} is {len(files)}')

final_files = 0

for filename in files:
    filepath = os.path.join(directory, filename)

    ds = xr.open_dataset(filepath)

    skip_file = False

    for variable in ds.data_vars:
        data = ds[variable]

        # check if all values are nan
        if np.isnan(data).all():
            print(f'All values in {variable} in file {filename} are NaN, skipping file...')
            skip_file = True
            break
        
        # check for partial NaN values
        if np.isnan(data).any():
            print(f'Found NaN values in {variable} in file {filename}, fixing...')
            ds[variable] = data.fillna(0)
            #ds[variable] = data.interpolate_na(dim=["y", "x"], method="nearest", use_coordinate=True)
        
        # check again for nan after interpolation
        #if np.isnan(ds[variable]).any():
                #print(f"Warning: Interpolation did not fill all NaN values in {variable} of {filename}.")
                # Optionally: Handle remaining NaNs (e.g., fill with a default value or skip the sample)
               # ds[variable] = ds[variable].fillna(0)    


    if skip_file == True:
        print(f'Processed file: {filename}')
        continue
    else:
        ds.to_netcdf(os.path.join(corrected_directory, f"corrected_{filename}"))
        final_files += 1
        print(f'Processed file: {filename}')


print(final_files)