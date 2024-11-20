import os
import netCDF4 as nc
import xarray as xr 
import matplotlib.pyplot as plt

output  = 'WildFireSpread/WildFireSpread_UNet3D/output_plots'
sample  = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_corrected/2007/corrected_sample_882.nc'

ds =  xr.open_dataset(sample)

print(ds.data_vars)
print(ds.dims)
data = ds['burned_areas']
#data = ds['ignition_points']


data2 = data.isel(time=slice(0, None))
#print(data2.data_vars)
data2 = data2.isel(time=4)


plt.figure(figsize=(10, 10))
data2.plot(cmap='viridis')
plt.title('test')
plt.savefig(output + '/' + 'test.png')
plt.show()

