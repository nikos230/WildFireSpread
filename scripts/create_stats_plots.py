import os
import pandas as pd
import os
import matplotlib.pyplot as plt



def calc_percentage(row):
    return ( row['samples'] / 9566 ) * 100


if __name__ == "__main__":
    os.system("clear")
    
    samples_per_year_path = 'output/samples_stats/samples_per_year.csv'
    samples_per_country_path = 'output/samples_stats/samples_per_country.csv'
    samples_per_month_path = 'output/samples_stats/samples_per_month.csv'

    df_year = pd.read_csv(samples_per_year_path, skiprows=1 ,header=None)
    df_year.rename(columns={0: 'year', 1: 'samples'}, inplace=True)
    df_year = df_year.sort_values(by='year', ascending=True)

    df_countries = pd.read_csv(samples_per_country_path, skiprows=1, header=None)
    df_countries.rename(columns={0: 'country', 1: 'samples'}, inplace=True)

    df_months = pd.read_csv(samples_per_month_path, skiprows=1, header=None)
    df_months.rename(columns={0: 'month_name', 1: 'month_index', 2: 'samples'}, inplace=True)
    

    # plot samples per year
    df_year.plot(x='year', y='samples', kind='bar', color='#1f77b4', legend=False, zorder=2)
    plt.title('Samples per Year 2006 - 2022', fontsize=10)
    plt.xlabel('Years', fontsize=8)
    plt.xticks(fontsize=7, color='black', rotation=0)
    plt.ylabel('Number of samples', fontsize=8)
    plt.yticks(fontsize=7.5)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)
    plt.text(-0.5, 1350, 'Total number of Samples: 9568', fontsize=6)
    plt.tight_layout()

    os.makedirs('output/samples_stats/plots', exist_ok=True)
    plt.savefig('output/samples_stats/plots/samples_per_year.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


    # plot samples per month
    df_months.plot(x='month_name', y='samples', kind='bar', color='#1f77b4', legend=False, zorder=2)
    plt.title('Samples per Month January - Decomber (2006 - 2022)', fontsize=10)
    plt.xlabel('Months', fontsize=8)
    plt.xticks(fontsize=7, color='black', rotation=30)
    plt.ylabel('Number of samples', fontsize=8)
    plt.yticks(fontsize=7.5)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)
    plt.tight_layout()

    os.makedirs('output/samples_stats/plots', exist_ok=True)
    plt.savefig('output/samples_stats/plots/samples_per_month.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


    # plot samples per country
    all_samples = df_countries['samples'].sum()


    df_countries['percentage'] = df_countries.apply(calc_percentage, axis=1)
    
    print(df_countries)
    


