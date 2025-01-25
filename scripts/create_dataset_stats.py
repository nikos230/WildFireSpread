import os
import xarray as xr
from datetime import datetime
import pandas as pd


if __name__ == '__main__':
    os.system("clear")

    dataset_path = '/home/n.anastasiou/nvme1/n.anastasiou/dataset_64_64_all_10days_final'


    fire_size = pd.DataFrame(columns=['sample', 'index', 'ha'])


    months = {
            'January':   {'index': 1,  'count': 0},
            'February':  {'index': 2,  'count': 0},
            'March':     {'index': 3,  'count': 0},
            'April':     {'index': 4,  'count': 0},
            'May':       {'index': 5,  'count': 0},
            'June':      {'index': 6,  'count': 0},
            'July':      {'index': 7,  'count': 0},
            'August':    {'index': 8,  'count': 0},
            'September': {'index': 9,  'count': 0},
            'October':   {'index': 10, 'count': 0},
            'November':  {'index': 11, 'count': 0},
            'December':  {'index': 12, 'count': 0}
            }

    country_dict = {}      # get samples per country
    year_dict = {}         # get samples per year
    number_of_samples = 0  # count all samples
    #years = ['2006']
    for year in os.listdir(dataset_path):
    #for year in years:
        dataset_path_year = os.path.join(dataset_path, year)
        for country in os.listdir(dataset_path_year):
            dataset_path_year_country = os.path.join(dataset_path_year, country)
            for sample in os.listdir(dataset_path_year_country):

                ds = xr.open_dataset(os.path.join(dataset_path_year_country, sample))

                country = ds.attrs['country']
                date = ds.attrs['date']
                date = datetime.strptime(date, '%Y-%m-%d')
                year = date.year
                month = date.month
                burned_area_ha = int(ds.attrs['burned_area_ha'])

                # get samples per country
                if country in country_dict:
                    country_dict[country] += 1
                    number_of_samples += 1
                else:
                    country_dict[country] = 1
                    number_of_samples += 1 

                # get samples per year
                if year in year_dict:
                    year_dict[year] += 1
                else:
                    year_dict[year] = 1
 

                # get samples per month
                for month_ in months:
                    if months[month_]['index'] == month:
                        months[month_]['count'] += 1

                # store burned area for later use, ['sample', 'index', 'ha']
                fire_size.loc[number_of_samples] = [sample, number_of_samples, burned_area_ha]

                ds.close()


    # convert dictionaries to Dataframes and then export them
    samples_per_year = pd.DataFrame.from_dict(year_dict, orient='index')
    samples_per_month = pd.DataFrame.from_dict(months, orient='index')
    samples_per_country = pd.DataFrame.from_dict(country_dict, orient='index')

    os.makedirs('output/samples_stats', exist_ok=True)            
    fire_size.to_csv('output/samples_stats/fire_sizes.csv')
    samples_per_country.to_csv('output/samples_stats/samples_per_country.csv')
    samples_per_year.to_csv('output/samples_stats/samples_per_year.csv')
    samples_per_month.to_csv('output/samples_stats/samples_per_month.csv')

    print(f'Samples per Country: {country_dict}')
    print(f'Samples per Year: {year_dict}')
    print(f'Samples per Month: {months}')
    print(f'Number of Total Samples: {number_of_samples}')