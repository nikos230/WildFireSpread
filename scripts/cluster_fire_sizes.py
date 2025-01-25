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

    fire_data = pd.read_csv('/home/n.anastasiou/hdd1/n.anastasiou/WildFireSpread/WildFireSpread_UNet3D/output/samples_stats/fire_sizes.csv')

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



    fire_data['manual_label'].value_counts().plot(kind='bar', color='red', zorder=2)
    plt.title('Fire Clusters', fontsize=12, color='black', loc='center')
    plt.locator_params(axis='y', nbins=12)
    plt.xlabel('Fire size (ha)', fontsize=8, color='black')
    plt.ylabel('Number of fires', fontsize=7.5, color='black')
    plt.xticks(fontsize=6, color='black', rotation=0)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=1)
    plt.tight_layout()
    os.makedirs('output/samples_stats/plots', exist_ok=True)
    plt.savefig('output/samples_stats/plots/fire_clusters.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


    # plt.figure(figsize=(8, 6))
    # for i in range(0, len(cluster_centers)):
    #     plt.scatter(
    #                 fire_size[labels == i],
    #                 np.zeros_like(fire_size[labels == i]),
    #                 s=5,
    #                 label=f'Cluster {i + 1}'
    #     )


    # plt.scatter(
    #     cluster_centers,
    #     np.zeros_like(cluster_centers),
    #     color='red',
    #     s=5,
    #     label='Cluster Centers',
    #     marker='X'
    # )   

    # plt.xlabel('Fire Size (ha)')
    # plt.yticks([])  # Hide the y-axis for better visualization
    # plt.savefig('output/test_plot.png', dpi=300) 