import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    os.system('clear')
    
    path_to_days = 'output/xAI/UNet3D/permute/days.csv'
    path_to_variables = 'output/xAI/UNet3D/permute/variables.csv'

    df_days = pd.read_csv(path_to_days, skiprows=0)
    df_variables = pd.read_csv(path_to_variables, skiprows=0)

    # zero all negative importnace values for variables
    df_copy = df_variables.copy()
    df_variables_negatives = df_variables[df_variables['importance'] < 0]
    #df_variables['importance'] = df_variables['importance'].clip(lower=0)
    df_variables = df_variables[df_variables['importance'] > 0]

    

    # zero all negative values for df_days
    df_days['importance'] = df_days['importance'].clip(lower=0)

    # remove igntion_points variable
    df_variables = df_variables[df_variables['variable'] != 'ignition_points']

    # group items
    df_variables = df_variables.groupby('variable')['importance'].mean().reset_index()
    df_days = df_days.groupby('day')['importance'].mean().reset_index()

    # sort values high to low in variable importance
    df_variables = df_variables.sort_values(by='importance', ascending=True)

    # rename variables
    df_variables['variable'] = df_variables['variable'].replace({'lc_sparse_vegetation': 'Sparse Vegatation (Static)',
                                              'lc_water_bodies': 'Land Cover: Water Bodies (Static)',
                                              'lc_wetland': 'Land Cover: Wetland (Static)',
                                              'lc_shrubland': 'Land Cover: Shrubland (Static)',
                                              'lc_settlement': 'Land Cover: Settlement (Static)',
                                              'tp': 'Total Percipitation (Dynamic)',
                                              'lc_grassland': 'Land Cover: Grassland (Static)',
                                              'slope': 'Slope (Static)',
                                              'lc_agriculture': 'Land Cover: Agriculture (Static)',
                                              'dem': 'Digital Elevation Model (Static)',
                                              'sp': 'Surface Pressure (Dynamic)',
                                              'smi': 'Soil Moisture Index (Dynamic)',
                                              'aspect': 'Aspect (Static)',
                                              'd2m': '2-meter Dew Point Temperature (Dynamic)',
                                              't2m': '2-meter Air Temperature (Dynamic)',
                                              'u': 'U-component of Wind (Dynamic)',
                                              'curvature': 'Curvature (Static)',
                                              'v': 'V-component of Wind (Dynamic)',
                                              'ssrd': 'Surface Solar Radiation Downwards (Dynamic)',
                                              'lc_forest': 'Land Cover: Forest (Static)',
                                              'ndvi': 'Normalized Difference Vegetation Index (Dynamic)',
                                              'lst_day': 'Land Surface Temperature Day (Dynamic)',
                                              'rh': 'Relative Humidity (Dynamic)',
                                              'lst_night': 'Land Surface Temperature Night (Dynamic)',
                                              'lai': 'Leaf Area Index (Dynamic)'})

    print(df_variables)
    print(df_days)

    # plot variable importance
    plt.barh(df_variables['variable'], df_variables['importance']*100, color="#0bb4ff", zorder=1) # plot positives importance
    plt.xlabel("Importance (%)")
    plt.ylabel("Variables")
    plt.yticks(fontsize=6)
    plt.title("Mean Importance of Variables")
    plt.grid(axis='x', linestyle=':', linewidth=0.5, color='gray', zorder=0)
    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_variables.png', dpi=400)
    plt.close()
    
    # plot days importance
    plt.bar(df_days['day'], df_days['importance']*100, color="#0bb4ff", zorder=1)
    plt.xlabel("Days")
    plt.xticks(range(0, 10, 1))
    plt.ylabel("Importance (%)")
    plt.title("Importance of Days")
    plt.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', zorder=0)

    plt.savefig('output/xAI/plots/xAI_days.png', dpi=400)
    plt.close()




    # Sorting by importance for better visualization
    df_copy['variable'] = df_copy['variable'].replace({'lc_sparse_vegetation': 'Sparse Vegatation (Static)',
                                                'lc_water_bodies': 'Land Cover: Water Bodies (Static)',
                                                'lc_wetland': 'Land Cover: Wetland (Static)',
                                                'lc_shrubland': 'Land Cover: Shrubland (Static)',
                                                'lc_settlement': 'Land Cover: Settlement (Static)',
                                                'tp': 'Total Percipitation (Dynamic)',
                                                'lc_grassland': 'Land Cover: Grassland (Static)',
                                                'slope': 'Slope (Static)',
                                                'lc_agriculture': 'Land Cover: Agriculture (Static)',
                                                'dem': 'Digital Elevation Model (Static)',
                                                'sp': 'Surface Pressure (Dynamic)',
                                                'smi': 'Soil Moisture Index (Dynamic)',
                                                'aspect': 'Aspect (Static)',
                                                'd2m': '2-meter Dew Point Temperature (Dynamic)',
                                                't2m': '2-meter Air Temperature (Dynamic)',
                                                'u': 'U-component of Wind (Dynamic)',
                                                'curvature': 'Curvature (Static)',
                                                'v': 'V-component of Wind (Dynamic)',
                                                'ssrd': 'Surface Solar Radiation Downwards (Dynamic)',
                                                'lc_forest': 'Land Cover: Forest (Static)',
                                                'ndvi': 'Normalized Difference Vegetation Index (Dynamic)',
                                                'lst_day': 'Land Surface Temperature Day (Dynamic)',
                                                'rh': 'Relative Humidity (Dynamic)',
                                                'lst_night': 'Land Surface Temperature Night (Dynamic)',
                                                'lai': 'Leaf Area Index (Dynamic)'})

    df_copy = df_copy[df_copy['variable'] != 'ignition_points']

    # df_positive = df_copy[df_copy["importance"] > 0].groupby("variable")["importance"].mean()
    # df_negative = df_copy[df_copy["importance"] < 0].groupby("variable")["importance"].mean()

    # df_combined = pd.concat([df_positive, df_negative], axis=1).fillna(0)
    # df_combined.columns = ["positive_mean", "negative_mean"]

    # df_combined["max_abs"] = df_combined.abs().max(axis=1)
    # df_combined = df_combined.sort_values(by="max_abs", ascending=True).drop(columns=["max_abs"])
    # # print(df_combined)
    # # print('\n')
    # # print(df_positive)
    # # print(df_negative)
    # variables = df_combined.index
    # y_pos = range(len(variables))

    # # Create horizontal bar plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.barh(y_pos, df_combined["positive_mean"], color='blue', label='Positive Mean')
    # ax.barh(y_pos, df_combined["negative_mean"], color='red', label='Negative Mean')

    # # Draw a vertical line at zero for separation
    # ax.axvline(0, color='black', linewidth=1)

    # # Labels and title
    # ax.set_xlabel("Importance")
    # ax.set_ylabel("Variable")
    # plt.yticks(fontsize=6)
    # ax.set_title("Feature Importance (Positive vs Negative Means)")
    # ax.set_yticks(y_pos)
    # ax.set_yticklabels(variables)
    # ax.legend() 

    # plt.tight_layout()
    # plt.savefig('output/xAI/plots/xAI_variables_test.png', dpi=400)
    # plt.close()



    # testing points
    df_copy["abs_importance"] = df_copy["importance"].abs()
    df_copy = df_copy.sort_values(by="abs_importance", ascending=True).drop(columns=["abs_importance"])

    # Prepare data for plotting
    variables = df_copy["variable"].unique()
    y_pos = {var: i for i, var in enumerate(variables)}

    # Create scatter plot in horizontal style
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df_copy.iterrows():
        ax.scatter(row["importance"]*100, y_pos[row["variable"]], color='#0bb4ff' if row["importance"] > 0 else '#e60049', marker='o', alpha=0.1, edgecolors='none', zorder=1)

    # Draw a vertical line at zero for separation
    ax.axvline(0, color='black', linewidth=1)

    # Labels and title
    ax.set_xlabel("Importance (%)")
    ax.set_ylabel("Variables")
    plt.yticks(fontsize=6)
    ax.set_title("Feature Importance (All Positive and Negative Values)")
    ax.set_yticks(range(len(variables)))
    ax.set_xticks(np.arange(-0.7*100, 0.8*100, 0.2*100))
    ax.set_yticklabels(variables)
    plt.grid(axis='x', linestyle=':', linewidth=0.5, color='gray', zorder=0)


    plt.tight_layout()
    plt.savefig('output/xAI/plots/xAI_variables_points.png', dpi=400)
    plt.close()

    
    # na dw posa samples einai positives kai posa negatives kathe plot

    
    