import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import glob

class BurnedAreaDataset(Dataset):
    def __init__(self, nc_files, transform=None):
        self.nc_files = nc_files   # .nc samples files
        self.transform = transform # 
        self.data = []             #
        self.load_data()           # load data from sample files

    def load_data(self):
        for file in self.nc_files:
            sample = xr.open_dataset(file)
            #sample = sample.isel(time=slice(None, 6)) 

            #print(sample)
            #exit()

            dynamic_vars = [
                'd2m', 
                'ignition_points', 
                'lai', 
                'lst_day', 
                'lst_night',
                'ndvi', 
                'rh', 
                'smi', 
                'sp', 
                'ssrd',
                't2m', 
                'tp',
                #'wind_direction',
                #'wind_direction_sin',
                #'wind_direction_cos',
                'u',
                'v',
                #'wind_speed'
            ]

            static_vars = [
                'aspect',
                'curvature',
                'dem', 
                #'roads_distance',
                'slope',
                'lc_agriculture',
                'lc_forest', 
                'lc_grassland', 
                'lc_settlement',
                'lc_shrubland', 
                'lc_sparse_vegetation',
                'lc_water_bodies',
                'lc_wetland', 
                #'population'
            ]

            time_steps = sample['time'].size # number of time steps in sample


            day4_ignition_points = sample['ignition_points'].isel(time=4)
            sample['ignition_points'][:] = 0 

            sample['ignition_points'].values[4, :, :] = day4_ignition_points
            
            #cnt = 0
            inputs = [] # list to put dynamic and static variables from sample
            # get dynamic variables from sample .nc file
            for variable in dynamic_vars:
                data_array = sample[variable].values
                inputs.append(data_array)
                #print(variable, cnt)
                #cnt += 1

            # get static variables from sample .nc file
            for variable in static_vars:
                data_array = sample[variable].values
                data_array = np.repeat(data_array[np.newaxis, :, :], time_steps, axis=0)
                inputs.append(data_array)
                #print(variable, cnt)
                #cnt += 1

            # put all varibles into a tensor
            input_tensor = np.stack(inputs, axis=0)    
            #print(input_tensor[0][0].shape)
            # normazise data, each channel individually
            for channel in range(input_tensor.shape[0]):
                # na ftia3w auto https://stackoverflow.com/questions/78008066/negative-values-in-normalized-data
                if channel == 12 or channel == 13:
                    channel_data = input_tensor[channel]
                    #min_value = channel_data.min()
                    #channel_data = channel_data + abs(min_value)

                    data_min = np.nanmin(channel_data)
                    data_max = np.nanmax(channel_data)
                    #input_tensor[channel] = -(channel_data - data_min) / (data_max - data_min)
                    if data_max - data_min > 0:
                        input_tensor[channel] = (channel_data - data_min) / (data_max - data_min)
                    else:
                        # same values all over the channel
                        input_tensor[channel] = input_tensor[channel]#np.zeros_like(channel_data)
                    
                else:    
                    #print(channel)
                    
                    channel_data = input_tensor[channel]
                    data_min = np.nanmin(channel_data)
                    data_max = np.nanmax(channel_data)
                    if data_max - data_min > 0:
                        input_tensor[channel] = (channel_data - data_min) / (data_max - data_min)
                    else:
                        # same values all over the channel
                        input_tensor[channel] = input_tensor[channel]#np.zeros_like(channel_data)   
                     

            # check for nan's and inf in input_tensor
            if np.isnan(input_tensor).any() or np.isinf(input_tensor).any():
                print('found value that is nan or inf!')
                continue

            # label varible
            label = sample['burned_areas'].values[4] # get label for time=4 (first fire day) 2
            label = (label > 0).astype(np.float32) # make is binary

            if np.isnan(label).any() or np.isinf(label).any():
                print('found nan or inf value on label!')
                continue
            
            self.data.append((input_tensor, label))
            sample.close()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, label = self.data[idx]

        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            input_tensor = self.transform(input_tensor)
            label = self.transform(label)
        return input_tensor, label    


if __name__ == '__main__':
    samples_path = ['WildFireSpread/WildFireSpread_UNET/dataset/dataset_sampled_corrected/2015/corrected_sample_0.nc']

    sample = BurnedAreaDataset(samples_path)

