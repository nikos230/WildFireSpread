import os
import geopandas as gpd



if __name__ == "__main__":
    os.system("clear")
    
    # ground truth path
    label_path = 'output/output_UNet3D/ground_truth_combined/ground_truth_combined.shp'


    # corine land cover 2018 paths
    corine_path = 'U2018_CLC2018_V2020_20u1.gpkg'
    #corine_greece_path = 'CLC_2012/CLC12_GR.shp'

    corine = gpd.read_file(corine_path, engine='pyogrio', use_arrow=True, layer='U2018_CLC2018_V2020_20u1')
    #corine = gpd.read_file(corine_greece_path, engine='pyogrio', use_arrow=True)
    corine = corine.to_crs('EPSG:4326')

    label = gpd.read_file(label_path, engine='pyogrio', use_arrow=True)

    clipped = gpd.clip(corine, label)

    clipped.to_file('output/corine_test/test_CLC.shp')

    