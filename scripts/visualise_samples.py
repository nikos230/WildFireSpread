import os
import netCDF4 as nc
import xarray as xr 
import matplotlib.pyplot as plt
from geogif import gif
import numpy as np
import imageio
import shutil


def interpolate(ds, variable):
    new_x = np.linspace(ds[variable].x.min(), ds[variable].x.max(), 512)
    new_y = np.linspace(ds[variable].y.min(), ds[variable].y.max(), 512)

    return ds[variable].interp(x=new_x, y=new_y, method='linear')



def plot_all_variables(ds, output_path):
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
        plt.savefig(f'{output_path}/variables_plots/variable_{variable}', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    return 0    



def plot_variables_single_plot(ds, dynamic_vars, static_vars, dynamic_vars_names, static_vars_names, output_path):
    rows = 2
    columns = 13

    x = list(range(1, 65, 10))
    y = list(range(1, 65, 10))

    figure, axis = plt.subplots(rows, columns, figsize=(columns * 1, rows * 1))

    dyn_cnt = 0
    for dynamic_variable in dynamic_vars:
        if dynamic_variable == 'ignition_points' or dynamic_variable == 'burned_areas' or dynamic_variable == 'spatial_ref' or dynamic_variable == 'wind_direction' or dynamic_variable == 'wind_speed' or dynamic_variable == 'wind_direction_sin' or dynamic_variable == 'wind_direction_cos':
            continue

        image_to_plot = axis[0, dyn_cnt].imshow(ds[dynamic_variable].values[4, :, :], cmap='viridis')
        axis[0, dyn_cnt].set_title(dynamic_vars_names[dyn_cnt], fontsize=4, pad=1.6)
        axis[0, dyn_cnt].set_xticks([])
        #axis[0, dyn_cnt].set_xticklabels(x, fontsize=2)
        axis[0, dyn_cnt].set_yticks([])
        #axis[0, dyn_cnt].set_yticklabels(y, fontsize=2)
        dyn_cnt += 1
        #figure.colorbar(image_to_plot, ax=axis[0, 0])
    #print(dyn_cnt)    



    stat_cnt = 0
    for static_variable in static_vars:
        image_to_plot = axis[1, stat_cnt].imshow(ds[static_variable].values[:, :], cmap='viridis')
        axis[1, stat_cnt].set_title(static_vars_names[stat_cnt], fontsize=4, pad=1.6)
        axis[1, stat_cnt].set_xticks([])
        #axis[1, stat_cnt].set_xticklabels(x, fontsize=2)
        axis[1, stat_cnt].set_yticks([])
        #axis[1, stat_cnt].set_yticklabels(y, fontsize=2)
        stat_cnt += 1
    #print(stat_cnt)

    axis[1, 12].axis("off")


    plt.subplots_adjust(wspace=0.001, hspace=0.2)
    #plt.tight_layout()
    plt.savefig(f'{output_path}/variables_single_plot.png', dpi=500, bbox_inches='tight')
    plt.close()

    return 0



def make_gif_dynamic_variables(ds, output, dynamic_vars, dynamic_vars_names, static_vars, static_vars_names):
    temp_path = f'{output}/temp'
    os.makedirs(temp_path, exist_ok=True)

    rows = 2
    columns = 13

    for time_step in range(0, 10):
        figure, axis = plt.subplots(rows, columns, figsize=(5 * 1, 1.1 * 1))
        #import matplotlib.font_manager as fm
        #available_fonts = sorted(f.name for f in fm.fontManager.ttflist)
        #print(available_fonts)
        #exit()
        
        dyn_cnt = 0
        for dynamic_variable in dynamic_vars:
            if dynamic_variable == 'ignition_points' or dynamic_variable == 'burned_areas' or dynamic_variable == 'spatial_ref' or dynamic_variable == 'wind_direction' or dynamic_variable == 'wind_speed' or dynamic_variable == 'wind_direction_sin' or dynamic_variable == 'wind_direction_cos':
                continue

            image_to_plot = axis[0, dyn_cnt].imshow(ds[dynamic_variable].values[time_step, :, :], cmap='viridis')
            axis[0, dyn_cnt].set_title(dynamic_vars_names[dyn_cnt], fontsize=3.8, pad=1.6)
            axis[0, dyn_cnt].set_xticks([])
            #axis[0, dyn_cnt].set_xticklabels(x, fontsize=2)
            axis[0, dyn_cnt].set_yticks([])
            #axis[0, dyn_cnt].set_yticklabels(y, fontsize=2)
            dyn_cnt += 1

        stat_cnt = 0
        for static_variable in static_vars:
                image_to_plot = axis[1, stat_cnt].imshow(ds[static_variable].values[:, :], cmap='viridis')
                axis[1, stat_cnt].set_title(static_vars_names[stat_cnt], fontsize=3.8, pad=1.6)
                axis[1, stat_cnt].set_xticks([])
                #axis[1, stat_cnt].set_xticklabels(x, fontsize=2)
                axis[1, stat_cnt].set_yticks([])
                #axis[1, stat_cnt].set_yticklabels(y, fontsize=2)
                stat_cnt += 1



        axis[1, 12].axis("off")
        figure.suptitle(f'Day - {time_step}', fontsize=6, y=0.15)
        plt.subplots_adjust(wspace=0.1, hspace=-0.2)
        plt.savefig(f'{temp_path}/time_step_{time_step}.jpg', dpi=1000, bbox_inches='tight', pad_inches=0.01)
        plt.close()

    time_steps = []
    for time_step in range(0, 10):
        time_step = f'time_step_{time_step}.jpg'
        path_to_time_step = os.path.join(temp_path, time_step)
        image = imageio.imread(path_to_time_step)
        time_steps.append(image)

    imageio.mimsave(f'{output}/variables.gif', time_steps, fps=1, loop=0, palettesize=256)    
    shutil.rmtree(temp_path)




if __name__ == "__main__":

    output  = 'output/visualise_samples'
    sample  = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final/2022/France/corrected_sample_10636.nc'

    os.makedirs(output, exist_ok=True)

    # open sample
    ds =  xr.open_dataset(sample)

    # clear all ignition points execpt day 4
    day4_ignition_points = ds['ignition_points'].isel(time=4)
    ds['ignition_points'].values[:] = 0 
    ds['ignition_points'].values[4, :, :] = day4_ignition_points

    print(ds.data_vars)
    print(ds.dims)

    dynamic_vars_names = [
            'Dewpoint Temp.', 
            #'ignition_points', 
            'LAI', 
            'LST - Day', 
            'LST - Night',
            'NDVI', 
            'Rel. Hum.', 
            'Soil Moist.', 
            'Surf. Press.', 
            'SSRD',
            'Air Temp.', 
            'Total Prec.',
            #'wind_direction',
            #'wind_direction_sin',
            #'wind_direction_cos',
            'u',
            'v'
            #'wind_speed'
        ]

    static_vars_names = [
            'Aspect',
            'Curvature',
            'DEM', 
            #'roads_distance',
            'Slope',
            'Agriculture',
            'Forest', 
            'Grassland', 
            'Settlement',
            'Shrubland', 
            'Spar. Veg.',
            'Water',
            'Wetland', 
            #'population'
        ]

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
            'v'
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
            'v'
            #'wind_speed'
        
        ]
        
    #plot_variables_single_plot(ds, dynamic_vars, static_vars, dynamic_vars_names, static_vars_names, output)
    #exit()

    make_gif_dynamic_variables(ds, output, dynamic_vars, dynamic_vars_names, static_vars, static_vars_names)
    exit()

    data = interpolate(ds, 'ndvi')
    gif = gif(data, to=f"{output}/test.gif", fps=1, date_size=20, date_position="lr") #, date_format=None
    exit()
 

    #plot_all_variables(ds, output_path)


