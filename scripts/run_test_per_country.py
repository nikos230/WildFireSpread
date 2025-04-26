import os
from ruamel.yaml import YAML
import pandas as pd
import subprocess


def edit_config(config, keys, country):

    for key in keys:
        if key in config['samples']:
            if key == 'test_countries':  
                config['samples'][key] = f'{country}'
    # reset fire sizes
    config['samples']['bunred_area_bigger_than'] = f'None'
    config['samples']['bunred_area_smaller_than'] = f'None'

    return config                



if __name__ == "__main__":
    os.system('clear')

    # path to dataset config
    path_to_config = 'configs/dataset.yaml'

    # path to available countries config
    path_to_countries = 'configs/available_countries.yaml'

    # path to fire clusters test results
    path_to_results_fire_clusters = 'output/evaluation_extra'
    os.makedirs(path_to_results_fire_clusters, exist_ok=True)

    # dataframe to keep results of test per cluster
    df_countries_test = pd.DataFrame(columns=['number_of_samples', 'dice', 'iou', 'precision', 'recall'])

    keys = [
            'test_countries'
            ]

    yaml = YAML()
    yaml.preserve_quotes = True  

    with open(path_to_config, "r") as file:
        dataset_config = yaml.load(file)
    file.close()

    # get countries list
    with open(path_to_countries, "r") as file:
        countries_config = yaml.load(file)
    file.close()   

    countries = []
    for country, value in countries_config['countries'].items():
        if value == 'True':
            countries.append(country)


    for country in countries:

        # open yaml file and edit it
        dataset_config = edit_config(dataset_config, keys, country)
        
        # write the new config file
        with open(path_to_config, "w") as file:
            yaml.dump(dataset_config, file,)
        file.close()    
        print(f'Dataset Config Updated!, new country: {country}')    

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

        df_countries_test.loc[country] = [number_of_test_samples, dice, iou, precision, recall]

    df_countries_test.to_csv(f'{path_to_results_fire_clusters}/countries_test_results.csv')    
    print('Done!!')

