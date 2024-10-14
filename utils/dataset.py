import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import normalize
from sklearn.model_selection import train_test_split

class NetCDFDataset(Dataset):
    def __init__(
        self,
        nc_file_paths,
        split='train',
        validation_split=0.2,
        test_split=0.1):



        # load the netcdf data
        self.nc_data = xr.open_dataset(nc_file_path)

        self.input_vars = [
            'd2m',
            't2m', 
            'wind_speed', 
            'lai',
            'lst_day', 
            'lst_night',
            'ndvi', 
            'rh', 
            'smi', 
            'sp', 
            'ssrd', 
            'tp'
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


        # load and process all netcdf files
        for nc_file_path in nc_file_paths:
            nc_data = xp.open(nc_file_path)
            self._process(nc_data)

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
        dynamic_inputs = np.stack(
            [normalize(nc_data[var].values) for var in self.input_vars], axis=1
        )

        static_inputs = np.stack(
            [normalize(nc_data[var].values) for var in self.static_vars], axis=0
        )
        static_inputs = np.repeat(static_inputs[np.newaxis, :, :, :], dynamic_inputs.shape[0], axis=0)

        inputs = np.concatenate([dynamic_inputs, static_inputs], axis=1)

        labels = normalize(nc_data[self.label_var].values[1:2, :, :])

        self.inputs.append(inputs)
        self.labels.append(labels)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tesnor(self.labels[idx], dtype=torch.float32)
