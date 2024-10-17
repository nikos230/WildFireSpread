import os
import xarray as xr
import numpy as np

# Directory where your .nc files are stored
directory = "WildFireSpread/dataset_small"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".nc"):
        filepath = os.path.join(directory, filename)
        
        # Load the .nc file
        try:
            ds = xr.open_dataset(filepath)
            print(f"Processing file: {filename}")
            
            # Loop through all variables in the dataset
            for var in ds.data_vars:
                data = ds[var]
                
                # Check for NaN values
                if np.isnan(data).any():
                    print(f"NaN values found in {var} of {filename}")
                    # Correct NaN values (e.g., replacing with zero)
                    ds[var] = data.fillna(0)
                
                # Check for Inf values
                if np.isinf(data).any():
                    print(f"Inf values found in {var} of {filename}")
                    # Correct Inf values (e.g., replacing with a large finite number)
                    ds[var] = xr.where(np.isinf(data), 0, data)
            
            # Save the corrected dataset
            corrected_filepath = os.path.join(directory, f"corrected_{filename}")
            ds.to_netcdf(corrected_filepath)
            print(f"Corrected file saved as {corrected_filepath}")
        
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
