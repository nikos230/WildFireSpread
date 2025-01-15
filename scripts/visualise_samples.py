import os
import netCDF4 as nc
import xarray as xr 
import matplotlib.pyplot as plt

output  = 'WildFireSpread/WildFireSpread_UNet3D/output_plots'
sample  = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final/2006/Bosnia & Herzegovina/corrected_sample_213.nc'

os.makedirs(output, exist_ok=True)

ds =  xr.open_dataset(sample)

print(ds.data_vars)
print(ds.dims)
#data = ds['burned_areas']
data = ds['ignition_points']

day4_ignition_points = ds['ignition_points'].isel(time=4)
ds['ignition_points'].values[:] = 0 

ds['ignition_points'].values[4 :, :] = day4_ignition_points

#data2 = data.isel(time=slice(4, None))
#print(data2.data_vars)
data2 = data.isel(time=4)


plt.figure(figsize=(10, 10))
data2.plot(cmap='viridis')
plt.title('test')
plt.savefig(output + '/' + 'test.png')
plt.show()

