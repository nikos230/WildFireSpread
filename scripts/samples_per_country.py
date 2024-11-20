import os
import xarray as xr



if __name__ == '__main__':
    os.system("clear")

    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_corrected_with_countries'

    country_dict = {}
    cnt = 0
    for year in os.listdir(dataset_path):
        dataset_path_year = os.path.join(dataset_path, year)
        for country in os.listdir(dataset_path_year):
            dataset_path_year_country = os.path.join(dataset_path_year, country)
            for sample in os.listdir(dataset_path_year_country):

                ds = xr.open_dataset(os.path.join(dataset_path_year_country, sample))
                country = ds.attrs['country']
                if country in country_dict:
                    country_dict[country] += 1
                    cnt += 1
                else:
                    country_dict[country] = 1
                    cnt += 1 
    print(country_dict, cnt)