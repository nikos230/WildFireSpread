import os
import xarray as xr

def check_nc_file_consistency(folder_path):
    # List all .nc files in the folder
    nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
    
    if not nc_files:
        print("No .nc files found in the folder.")
        return
    
    # Initialize reference variables and dimensions
    reference_data_vars = None
    reference_dims = {'x': 64, 'y': 64, 'time': 4}

    
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
                print(f"Data variables mismatch in file: {nc_file}")
                print(f"Expected: {reference_data_vars}, but got: {current_data_vars}")
            
            # Check if the current file has the same dimensions
            if current_dims != reference_dims:
                print(f"Dimension mismatch in file: {nc_file}")
                print(f"Expected: {reference_dims}, but got: {current_dims}")

    print("Consistency check completed.")

# Example usage
folder_path = 'WildFireSpread/train_dataset'
check_nc_file_consistency(folder_path)
