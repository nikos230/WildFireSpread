import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import normalize


class NetCDFDataset(Dataset):
    def __init__(self, nc_file_path):
        # load the netcdf data
        self.nc_data = xr.open_dataset(nc_file_path)

        self.input_vars = ['aspect',
                            ]
