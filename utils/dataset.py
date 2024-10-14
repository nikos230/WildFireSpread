import os
import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDF4Dataset  # Import the correct Dataset class

class NetCDFDataset(Dataset):
    def __init__(self, file_list, split='train', transform=None):
        """
        Args:
            file_list (list): List of .nc file paths.
            split (str): Specify 'train', 'validation', or 'test' to set the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = file_list
        self.split = split
        self.transform = transform

        # Define input and static variables
        self.input_vars = [
            'aspect', 'curvature', 'd2m', 'dem', 'lai',
            'lst_day', 'lst_night', 'ndvi', 'rh', 'slope',
            'smi', 'sp', 'ssrd', 't2m', 'tp',
            'wind_direction', 'wind_speed', 'population'
        ]
        self.static_vars = [
            'lc_agriculture', 'lc_forest', 'lc_grassland', 
            'lc_settlement', 'lc_shrubland', 'lc_sparse_vegetation', 
            'lc_water_bodies', 'lc_wetland', 'roads_distance'
        ]

        # Split logic
        total_files = len(self.file_list)
        if self.split == 'train':
            self.file_list = self.file_list[:int(total_files * 0.7)]
        elif self.split == 'validation':
            self.file_list = self.file_list[int(total_files * 0.7):int(total_files * 0.85)]
        elif self.split == 'test':
            self.file_list = self.file_list[int(total_files * 0.85):]
        else:
            raise ValueError("Invalid split type. Choose 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the .nc file
        file_path = self.file_list[idx]
        nc_file = NetCDF4Dataset(file_path)

        features = []


        # Load static variables (2D)
        for var in self.static_vars:
            data = nc_file.variables[var][:]
            if data.ndim != 2:
                raise ValueError(f"Static Variable '{var}' does not have the expected shape: {data.shape}")
            
            # Expand dimension to add time steps (4 in your case)
            data = np.tile(data[np.newaxis, ...], (4, 1, 1))  # Shape becomes (4, 64, 64)
            features.append(data)

        # Load dynamic variables (3D)
        for var in self.input_vars:
            data = nc_file.variables[var][:]
            print(f"Variable '{var}' has shape {data.shape} and dtype {data.dtype}")  # Debugging output
        
            if data.ndim == 3:
                features.append(data)
            elif data.ndim == 2:
                data = np.expand_dims(data, axis=0)  # (1, 64, 64)
                data = np.tile(data, (4, 1, 1))  # Now shape is (4, 64, 64)
                features.append(data)
            else:
                raise ValueError(f"Variable '{var}' does not have the expected shape: {data.shape}")

        # Check shapes of features
        for i, feature in enumerate(features):
            print(f"Feature {i} shape: {feature.shape}")

        # Concatenate features
        features_tensor = torch.from_numpy(np.concatenate(features, axis=0)).float()
        
        # Load target variable (assuming it also has the time dimension)
        target = nc_file.variables['burned_areas'][:]
        target_tensor = torch.from_numpy(target).float()

        # Optional transform
        if self.transform:
            features_tensor = self.transform(features_tensor)
            target_tensor = self.transform(target_tensor)

        return features_tensor, target_tensor



