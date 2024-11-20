import os
import xarray as xr
import geopandas as gpd



if __name__ == '__main__':
    os.system("clear")
    # dataset path
    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_corrected_with_countries_new'
    # load shapefiles
    predicted = gpd.read_file('WildFireSpread/WildFireSpread_UNet3D/out_shapefiles/predicted_combined/predicted_combined.shp')
    label = gpd.read_file('WildFireSpread/WildFireSpread_UNet3D/out_shapefiles/ground_truth_combined/ground_truth_combined.shp')

    merged = predicted.merge(label, on='sample', suffixes=('_predicted', '_label'))

    merged['diff'] = merged['b_area_ha_predicted'] - merged['b_area_ha_label']

    low = 0         # 0 - 100
    moderate = 0    # 100 - 300
    high = 0        # 300 - 600
    very_high = 0   # 600 - 900
    ultra_sport = 0 # 900 - inf

    for i in range(len(merged)):
        predicted_burned = merged['b_area_ha_predicted'].iloc[i]
        label_burned = merged['b_area_ha_label'].iloc[i]
        diff = merged['diff'].iloc[i]
        dice = merged['dice_predicted'].iloc[i]

        if diff < 100:
            low += 1
        elif diff > 100 and diff < 300:
            moderate += 1
        elif diff > 300 and diff < 600:
            high += 1
        elif diff > 600 and diff < 900:
            very_high += 1
        elif diff > 900:
            ultra_sport += 1                

        print(f'Predicted: {round(predicted_burned, 0)}, label: {label_burned}, dice: {round(dice, 2)}, diff: {round(abs(diff), 0)}')
    print(merged['diff'].sum()/len(merged))

    print(f'low: {low}, moderate: {moderate}, high: {high}, very high: {very_high}, ultra: {ultra_sport}\nTotal: {len(merged)}')   