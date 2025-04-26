import os
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
from shapely.geometry import Point
from shapely.wkt import loads

# shapefile date format : year-month-day

if __name__ == '__main__':
    os.system('clear')

    shapefile_path = 'hdd1/n.anastasiou/mesogeos/ml_tracks/b.final_burned_area/burned_areas_shapefile/med_burned_areas_updated.shp'
    
    gf = gpd.read_file(shapefile_path, engine='pyogrio', use_arrow=True)

    start_date = gf['IGNITION_D']
    end_date = gf['LAST_DATE']
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # select year 2006 to 2022
    start_date = start_date[start_date >= pd.Timestamp('2006-01-01')]
    end_date = end_date[end_date >= pd.Timestamp('2006-01-01')]
    

    diff = end_date - start_date

    days = diff.dt.days
    days = days[days <= 10]
    
    plt.hist(days, bins=days.max(), edgecolor='black')
    plt.title('')
    plt.xlabel('Duration of fire in Days', fontsize=8)
    plt.ylabel('Number of fires', fontsize=8)
    plt.xticks(range(0, days.max() + 1))
    plt.savefig('hdd1/n.anastasiou/WildFireSpread/WildFireSpread_UNet3D/output/histogram.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    exit()

    # create histogram for extracted dataset
    dataset_path = 'nvme1/n.anastasiou/dataset_64_64_all_10days_final'

    # ignition points from burned areas shapefile
    #ignition_points = gf['geometry_h']
    gf['geometry_h'] = gf['geometry_h'].apply(loads)
    gf = gpd.GeoDataFrame(gf, geometry='geometry_h', crs="EPSG:5643")  # Replace with your CRS
    
    gf = gf.to_crs('EPSG:4326')

    new_gf = gpd.GeoDataFrame(columns=gf.columns, crs=gf.crs)

    for year in os.listdir(dataset_path):
        dataset_path_year = os.path.join(dataset_path, year)
        for country in os.listdir(dataset_path_year):
            dataset_path_year_country = os.path.join(dataset_path_year, country)
            for sample in os.listdir(dataset_path_year_country):

                #ds = xr.open_dataset(os.path.join(dataset_path_year_country, sample))
                ds = xr.open_dataset('/home/n.anastasiou/nvme1/n.anastasiou/dataset_corrected_new/2006/Bulgaria/corrected_corrected_sample_125.nc')
                ignition_point = ds['ignition_points'].isel(time=4)

                # reproject variable into EPSG:5643
                ignition_point = ignition_point.rio.write_crs("EPSG:4326")
                #ignition_point = ignition_point.rio.reproject("EPSG:5643")

                ignition_point = ignition_point.where(ignition_point > 0, drop=True)
                #print(ignition_point)
                #exit()
                ignition_points_coords = ignition_point.coords['x'], ignition_point.coords['y']
                
                point = Point(round(float(ignition_points_coords[0].values[0]), 4), round(float(ignition_points_coords[1].values[0]), 4))

                matched_row = gf[gf['geometry_h'] == point] #& (gf['IGNITION_D'] == ds.attrs['date'])]
 

                if matched_row.empty:
                    #gf['distance'] = gf[gf['IGNITION_D'] == ds.attrs['date']]['geometry_h'].distance(point)
                    gf['distance'] = gf['geometry_h'].distance(point)
                    nearest_row = gf.loc[gf['distance'].idxmin()]
                    new_gf = new_gf.append(nearest_row, ignore_index=True) 
                    print(nearest_row['geometry_h'])
                    print(ds)
                    print(nearest_row)
                else:
                    new_gf = new_gf.append(matched_row, ignore_index=True)

                    
                #print(matched_row)
                print(ignition_points_coords[0].values, ignition_points_coords[1].values)
                print(sample)
                exit()


