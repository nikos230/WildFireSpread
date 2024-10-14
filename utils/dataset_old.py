import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.utils import normalize
from sklearn.model_selection import train_test_split
import glob
import sys
import os
import logging

class NetCDFDataset(Dataset):
    def __init__(
        self,
        directory,
        split='train',
        validation_split=0.2,
        test_split=0.1):

        self.input_vars = [
            'd2m',
            't2m',
            'ignition_points', 
            'wind_speed',
            'wind_direction', 
            'lai',
            'lst_day', 
            'lst_night',
            'ndvi', 
            'rh', 
            'smi', 
            'sp', 
            'ssrd', 
            'tp',
        ]

        self.static_vars = [
            'aspect', 
            'curvature', 
            'dem', 
            'roads_distance', 
            'slope',
            'lc_agriculture', 
            'lc_forest', 
            'lc_grassland', 
            'lc_settlement',
            'lc_shrubland', 
            'lc_sparse_vegetation', 
            'lc_water_bodies', 
            'lc_wetland',
            'population'
        ]


        # label 
        self.label_var = 'burned_areas'

        self.inputs = []
        self.labels = []

        def get_nc_files(directory):
            nc_file_paths = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.nc'):
                        nc_file_paths.append(entry.path)
            return nc_file_paths
        nc_file_paths = get_nc_files('WildFireSpread/dataset_small')

        # load and process all netcdf files
        for nc_file_path in nc_file_paths:
            nc_data = xr.open_dataset(nc_file_path, engine='netcdf4')
            self._process_file(nc_data)

        if not self.inputs or not self.labels:
            raise ValueError("No data found in the specified NetCDF files.")

        self.inputs = np.concatenate(self.inputs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # split data into training, validation and test sets
        train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(self.inputs, self.labels, test_size=(validation_split + test_split), random_state=42)
        validation_size = validation_split / (validation_split + test_split)
        validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=validation_size, random_state=42)

        if split == 'train':
            self.inputs, self.labels = train_inputs, train_labels
        elif split == 'validation':
            self.inputs, self.labels = validation_inputs, validation_labels
        else:
            raise ValueError("split must be 'train, 'validation' or 'test'")


    def _process_file(self, nc_data):
        dynamic_inputs = []
        for var in self.input_vars:
            if var in nc_data:
                dynamic_inputs.append(normalize(nc_data[var].values))
            else:
                logging.warning(f"Dynamic variable '{var}' is missing. Filling with zeros.")
                dynamic_inputs.append(np.zeros_like(nc_data[self.input_vars[0]].values))

        dynamic_inputs = np.stack(dynamic_inputs, axis=0)

        static_inputs = []
        for var in self.static_vars:
            if var in nc_data:
                static_inputs.append(normalize(nc_data[var].values))
            else:
                logging.warning(f"Static variable '{var}' is missing. Filling with zeros.")
                static_inputs.append(np.zeros_like(nc_data[self.static_vars[0]].values))

        static_inputs = np.stack(static_inputs, axis=0)
        static_inputs = np.repeat(static_inputs[np.newaxis, :, :, :], dynamic_inputs.shape[0], axis=0)

        inputs = np.concatenate([dynamic_inputs, static_inputs], axis=1)

        # Extract label at time=0
        if self.label_var in nc_data:
            labels = normalize(nc_data[self.label_var].values[0:1, :, :])
        else:
            logging.warning(f"Label variable '{self.label_var}' is missing in file. Filling with zeros.")
            labels = np.zeros((1, nc_data.dims['y'], nc_data.dims['x']))  # Adjust dimensions accordingly

        # Append processed inputs and labels
        print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
        self.inputs.append(inputs)
        self.labels.append(labels)


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tesnor(self.labels[idx], dtype=torch.float32))
