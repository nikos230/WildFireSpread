import os
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt

output  = 'WildFireSpread/WildFireSpread_UNET/output_plots'
sample  = 'WildFireSpread/dataset_mesegeos/2021/sample_0.nc'

ds =  xr.open_dataset(sample)

print(ds.data_vars)
print(ds.dims)
data = ds['burned_areas']
exit()

aspect = ds['aspect']
print(aspect)
exit()
data2 = data.isel(time=0)
#print(data2.dims)
#print(data['time'].values)

plt.figure(figsize=(10, 10))
data2.plot(cmap='viridis')
plt.title('test')
plt.savefig(output + 'test.png')
plt.show()

