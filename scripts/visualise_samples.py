import os
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt

output  = 'WildFireSpread/WildFireSpread_UNET/output_plots'
sample  = 'WildFireSpread/WildFireSpread_UNET/dataset/dataset_sampled/2015/sample_0.nc'

ds =  xr.open_dataset(sample)

print(ds.data_vars)
print(ds.dims)
data = ds['burned_areas']
#data = ds['ignition_points']



data2 = data.isel(time=2)
print(data2.dims)


plt.figure(figsize=(10, 10))
data2.plot(cmap='viridis')
plt.title('test')
plt.savefig(output + '/' + 'test.png')
plt.show()

