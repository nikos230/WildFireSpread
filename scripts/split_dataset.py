import os
import random
import shutil
import yaml


def split_dataset(dataset_path, train_dataset_path, test_dataset_path, num_of_samples, train_split_per, test_split_per):
    # make directories if not exist already
    os.makedirs(train_dataset_path, exist_ok=True)
    os.makedirs(test_dataset_path, exist_ok=True)

    # get all sample files from dataset path and put them into a list (names of files)
    sample_files = [file for file in os.listdir(dataset_path) if file.endswith('.nc')]

    if num_of_samples > len(sample_files):
        print(f'Max number of avaible samples: {len(sample_files)}')
        exit()
        
    sample_files = random.sample(sample_files, num_of_samples)
    random.shuffle(sample_files)



    split_index = int(train_split_per * len(sample_files))

    train_samples = sample_files[:split_index]
    test_samples = sample_files[split_index:]

    # copy files from dataset_path to train_dataset_path and test_dataset_path
    for train_sample in train_samples:
        shutil.copy(os.path.join(dataset_path, train_sample), os.path.join(train_dataset_path, train_sample))

    for test_sample in test_samples:
        shutil.copy(os.path.join(dataset_path, test_sample), os.path.join(test_dataset_path, test_sample))

    print(f'\n Done! \n Total train samples: {len(train_samples)} \n Total test samples: {len(test_samples)}')        



if __name__ == '__main__':
    with open('WildFireSpread/WildFireSpread_UNET/configs/split_dataset.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    config_file.close()

    dataset_path       = config['dataset']['dataset_path']
    train_dataset_path = config['dataset']['train_dataset_path']
    test_dataset_path  = config['dataset']['test_dataset_path']

    num_of_samples     = int(config['samples']['numbner_of_samples'])
    train_split_per    = float(config['samples']['train_split'])
    test_split_per     = 1 - float(train_split_per)
    
    print(f'Settings for spliting the Dataset: \n Train dataset save path: {train_dataset_path} \n Test dataset save path: {test_dataset_path} \n\n Total number of samples: {num_of_samples} \n Train split: {train_split_per} \n Test split: {test_split_per:.1f}')


    split_dataset(dataset_path, train_dataset_path, test_dataset_path, num_of_samples, train_split_per, test_split_per)