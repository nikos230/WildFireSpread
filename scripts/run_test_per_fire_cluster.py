import os
from ruamel.yaml import YAML
import pandas as pd
import subprocess


def edit_config(config, keys, min, max):

    for key in keys:
        if key in config['samples']:
            if key == 'bunred_area_bigger_than':
                config['samples'][key] = f'{min}'
            elif key == 'bunred_area_smaller_than':
                config['samples'][key] = f'{max}'  
    # reset countries
    config['samples']['test_countries'] = f'all'             
    return config                



if __name__ == "__main__":
    os.system('clear')

    # path to dataset config
    path_to_config = 'configs/dataset.yaml'

    # path to fire clusters test results
    path_to_results_fire_clusters = 'output/evaluation_extra'
    os.makedirs(path_to_results_fire_clusters, exist_ok=True)

    # dataframe to keep results of test per cluster
    df_cluster_test = pd.DataFrame(columns=['number_of_samples', 'min', 'max', 'dice', 'iou', 'precision', 'recall'])

    keys = [
            'bunred_area_bigger_than',
            'bunred_area_smaller_than'
            ]

    yaml = YAML()
    yaml.preserve_quotes = True  

    with open(path_to_config, "r") as file:
        dataset_config = yaml.load(file)


    fire_clusters = {'cluster_1': {'min': 0,     'max': 1500},
                     'cluster_2': {'min': 1500,  'max': 3000},
                     'cluster_3': {'min': 3000,  'max': 5000},
                     'cluster_4': {'min': 5000,  'max': 8500},
                     'cluster_5': {'min': 8500,  'max': 14000},
                     'cluster_6': {'min': 14000, 'max': 20000},
                     'cluster_7': {'min': 20000, 'max': 35000},
                     'cluster_8': {'min': 35000, 'max': 5000000}
                     }


    for cluster in fire_clusters:
        min_size = int(fire_clusters[cluster]['min'])
        max_size = int(fire_clusters[cluster]['max'])

        # open yaml file and edit it
        dataset_config = edit_config(dataset_config, keys, min_size, max_size)
        
        with open(path_to_config, "w") as file:
            yaml.dump(dataset_config, file)
        print(f'Dataset Config Updated!, new fire sizes min:{min_size}, max:{max_size}')    
        exit()
        # run the test script and keep the results
        result = subprocess.run(
                                ['python', 'test_unet3d.py'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                                )
        output = result.stdout.decode('utf-8')

        number_of_test_samples = 0
        dice = 0
        iou = 0
        precision = 0
        recall = 0

        for line in output.split('\n'):

            if 'Number of test samples' in line:
                number_of_test_samples = line.split(':')[-1].strip() 
            elif 'Dice Coefficient' in line:
                dice = line.split(':')[-1].strip()   
            elif 'IoU' in line:
                iou = line.split(':')[-1].strip() 
            elif 'Precision' in line:
                precision = line.split(':')[-1].strip()          
            elif 'Recall' in line:
                recall = line.split(':')[-1].strip()                                     

        df_cluster_test.loc[cluster] = [number_of_test_samples, min_size, max_size, dice, iou, precision, recall]

        
    df_cluster_test.to_csv(f'{path_to_results_fire_clusters}/fire_clusters_test_results.csv')    
    print('Done!!')

