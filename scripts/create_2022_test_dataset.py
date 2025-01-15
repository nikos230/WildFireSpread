import os
import xarray as xr
from datetime import datetime
import shutil


if __name__ == "__main__":

    path_to_dataset = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final'
    path_to_test_dataset = '/home/n.anastasiou/nvme1/n.anastasiou/test_2022'

    year = str(2022)    

    path_year = os.path.join(path_to_dataset, year)
    for country in os.listdir(path_year):
        path_country = os.path.join(path_year, country)
        for sample in os.listdir(path_country):
            path_to_sample = os.path.join(path_country, sample)
            ds = xr.open_dataset(path_to_sample)

            date = str(ds.attrs['date'])
            date = datetime.strptime(date, "%Y-%m-%d")

            month = date.month
            year_ = date.year

            if month >= 6 and year_ == 2022:
                continue
            else:
                shutil.move(path_to_sample, path_to_test_dataset)
                
    
    