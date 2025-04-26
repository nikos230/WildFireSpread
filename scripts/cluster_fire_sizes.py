import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



def assing_label(row):

    if row['ha'] > 0 and row['ha'] <= 1500:
        return '250 - 1500'
    
    elif row['ha'] > 1500 and row['ha'] <= 3000:
        return '1500 - 3000'

    elif row['ha'] > 3000 and row['ha'] <= 5000:
        return '3000 - 5000'    

    elif row['ha'] > 5000 and row['ha'] <= 8500:
        return '5000 - 8500'  

    elif row['ha'] > 8500 and row['ha'] <= 14000:
        return '8500 - 14000'  

    elif row['ha'] > 14000 and row['ha'] <= 20000:
        return '14000 - 20000'      

    elif row['ha'] > 20000 and row['ha'] <= 35000:
        return '20000 - 35000'  

    elif row['ha'] > 35000:
        return '> 35000'         



if __name__ == "__main__":
    os.system("clear")

    fire_data = pd.read_csv('output/samples_stats/fire_sizes.csv')

    fire_size = fire_data['ha'].to_numpy()

    num_clusters = 10
    km = KMeans(num_clusters, init='k-means++')

    labels = km.fit_predict(fire_size.reshape(-1, 1))
    cluster_centers = km.cluster_centers_
    

    fire_data['kmeans_label'] = labels


    minmax_values = fire_data.groupby('kmeans_label')['ha'].agg(['min', 'max'])
    print(minmax_values)
    
    minmax_mapping = minmax_values.apply(lambda x: f"{x['min']}-{x['max']}", axis=1).to_dict()
    print(minmax_mapping)

    fire_data['kmeans_label'] = fire_data['kmeans_label'].map(minmax_mapping)


    # minmax_mapping = {
    #     0: '5100-8300',  check
    #     1: '1500-2800',  check
    #     2: '35400-52500',
    #     3: '13500-21000', check
    #     4: '105803-139655',
    #     5: '56961-85192', 
    #     6: '21300-34100', check
    #     7: '8494-13366', check
    #     8: '2900-5000',  check
    #     9: '250-1400'     check 
    #     }

    fire_data['manual_label'] = fire_data.apply(assing_label, axis=1)
    #exit()



    fire_data['manual_label'].value_counts().plot(kind='bar', color='#e60049', zorder=2)
    plt.title('Fire Clusters', fontsize=12, color='black', loc='center')
    plt.locator_params(axis='y', nbins=12)
    plt.xlabel('Fire size (ha)', fontsize=10, color='black')
    plt.yticks(fontsize=9)
    plt.ylabel('Number of fires', fontsize=10, color='black')
    plt.xticks(fontsize=6, color='black', rotation=0)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)
    plt.tight_layout()
    os.makedirs('output/samples_stats/plots', exist_ok=True)
    plt.savefig('output/samples_stats/plots/fire_clusters.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()



    try:
        fire_clustres_results = pd.read_csv('output/evaluation_extra/fire_clusters_test_results.csv')
    except FileNotFoundError:
        print('fire clusters results not found! if you want run "run_test_per_fire_cluster.py" !!')
        exit(2)

    fire_clustres_results['label'] = fire_clustres_results['min'].astype(str) + ' - ' + fire_clustres_results['max'].astype(str)
    fire_clustres_results.at[0, 'label'] = '150 - 1500'
    fire_clustres_results.at[7, 'label'] = '> 35000'
    print(fire_clustres_results)
    
    fire_clustres_results = fire_clustres_results.sort_values(by='dice', ascending=False)


    # plot both the fire clusters results and the fire clusters
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    # plot fire clusters results first
    fire_clusters_data = fire_data['manual_label'].value_counts()
    bars1 = ax1.bar(fire_clustres_results['label'], fire_clustres_results['dice'] * 100, color='#0bb4ff', width=0.4, label='Dice / F1 Score', zorder=1, align='center')
    ax1.set_xlabel('Fire size (ha)', fontsize=10, color='black')
    ax1.set_ylabel('Dice / F1 Score (%)', fontsize=9, color='black')
    ax1.tick_params(axis='y', labelsize=9)
    ax1.set_xticklabels(fire_clustres_results['label'], fontsize=6, color='black')
    #ax1.grid(axis='y', linestyle=':', linewidth=0.5, color='blue', zorder=1)


    # plot fire clusters
    
    
    bars2 = ax2.bar(fire_clustres_results['label'], fire_clusters_data.values, color='#e60049', width=0.4, align='edge', zorder=4, label='Fire Cluster')
    ax2.set_ylabel('Number of Fires', fontsize=10, color='black')
    ax2.tick_params(axis='y', labelsize=9, colors='black')

    for bar in bars1:
        height_of_bar = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height_of_bar, f'{height_of_bar:.1f}', ha='center', va='bottom', fontsize=7, color='black')


    plt.title('Dice / F1 Score per Fire Cluster', fontsize=12, color='black')
    #plt.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    os.makedirs('output/evaluation_extra/plots', exist_ok=True)
    plt.savefig('output/evaluation_extra/plots/fire_clusters_test_results_combined.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
