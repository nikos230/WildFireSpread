import os
import netCDF4 as nc
import xarray as xr 
import matplotlib.pyplot as plt

output  = 'output_plots'
sample  = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final/2022/France/corrected_sample_10636.nc'

os.makedirs(output, exist_ok=True)

ds =  xr.open_dataset(sample)

print(ds.data_vars)
print(ds.dims)
data = ds['burned_areas']
#data = ds['ignition_points']

day4_ignition_points = ds['ignition_points'].isel(time=4)
ds['ignition_points'].values[:] = 0 

ds['ignition_points'].values[4 :, :] = day4_ignition_points

#data2 = data.isel(time=slice(4, None))
#print(data2.data_vars)
data2 = data.isel(time=4)


plt.figure(figsize=(3, 3))
#data2.plot(cmap='gray')
plt.imshow(data2, cmap='gray')
#plt.title('test')
plt.axis('off')
plt.savefig(output + '/' + 'test.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()
#exit()

for variable in ds.data_vars:
    if variable == 'burned_areas' or variable == 'ignition_points' or variable == 'spatial_ref':
        continue
    print(f'Saving...{variable}')
    print(f'Shape {ds[variable].shape}') 
    if len(ds[variable].shape) <= 2:
        if ds[variable].shape[0] == 64 and ds[variable].shape[1] == 64:
            variable_to_plot = ds[variable]
    else:
        variable_to_plot = ds[variable].values[4]
        

    plt.imshow(variable_to_plot, cmap='gray')
    plt.axis('off')
    plt.savefig(f'output_plots/plots2/variable_{variable}', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    

