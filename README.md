<div align="center">
  <img src="https://raw.githubusercontent.com/nikos230/WildFireSpread/main/logos/fire.png" alt="Fire Logo" />
</div>
<br /> 
(work in progress... as of Feb 2025)



# About The Project
A Machine Learning ready Dataset and Model to predict final burned area from a Wildfire. <br /> <br /> 
This project is part of my Thesis and makes use of [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) Dataset and UNet models to predict final burned area from 27 remote sensing variables using spatial and temporal data.<br /> <br /> 
**This project includes** :
- Deep Learing Models 
- Dataset with ≈ 9500 samples
- Tools for model evalution and visualization of results in shapefile form
- Tools for Dataset statistics extraction
- Showcase of results
<br />

## Dataset
Training and Testing of the Models have been done with [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) Dataset. Training data (and validation) are 27 variables in 64 x 64 km paches and a Spatial Resolution of 1 pixel = 1km x 1km in netCDF format. The mesogeos DataCube has 1km x 1km x 1day Spatial and Temporal resolution of 1 day. <br /> <br />
The Samples from the DataCube have 64km x 64km x 1day resoltuion, for every fire event there is a 64 x 64km patch around the fire with a random offset so the burned area is not always in the middle of the patch, and for every sample there are 10 days in total, 5 before the fire started and 5 days after. Samples are from the mesogeos region and includes ≈9500 samples from 31 countries and from years 2006 to 2022. <br /><br />
![Alt text](https://raw.githubusercontent.com/nikos230/WildFireSpread/main/screenshots/dynamic_variables.gif)
<br /><br />


## Deep Learning Models
This project makes use of UNet2D and UNet3D models, the main difference in UNet3D is the 3D convolution which takes into account the temporal information of the samples as the 3D convolution can get info from 3 or more days at once, the 2D convolution is good for spartial feature extraction but not for temporal feature extraction. <br />
In Feature work the Dataset will be tested on a Vision Transformer (ViT) and results will be published. <br /><br />
<br />

## Models Evaluation and Metrics (latest results Feb 2025)
The main evaluation metric is the Dice Coefficient, presented below, which shows how much 2 shapes are similar, in this Segmatation Task along with Intersection Over Union (IoU) are the most important metrics.
A baseline UNet2D is trained on only the fire day and then with all samples meaning using all 10 days each sample has. 


| Metrics (%)         | UNet2D<br /> Baseline (1 day) | UNet2D<br /> 10days  | UNet3D <br /> 10days  | UNet3D <br /> 5 days after fire 
|:------------------:|:------------------------------:|:--------------------:|:---------------------:|:--------------------------:|
| F1 Score / Dice    | 52.7                           | 56.1                 | **57.2**              | 57.3                  
| IoU                | 37.1                           | 40.7                 | 41.9                  | 42.1
| Precision          | 51.0                           | 58.2                 | 59.0                  | 60.5
| Recall             | 68.0                           | 64.0                 | 64.8                  | 63.1

<br />

| Metrics (%)      | UNet3D <br /> 9days  | UNet3D <br /> 8days | UNet3D <br /> 7days | UNet3D <br /> 6days
|:----------------:|:-------------------:|:-------------------:|:-------------------:|:------------------:|
| F1 Score / Dice  | 55.1                | 54.0                | 54.1                | 53.2
| IoU              | 39.8                | 38.7                | 38.8                | 37.9
| Precision        | 56.2                | 55.2                | 55.2                | 54.8
| Recall           | 64.7                | 64.1                | 64.5                | 63.1

<br />

- Baseline Model is trained using only the fire day with 27 variables <br />
- UNet2D is trained using all spatial and temporal data using 117 variables which are the 10 days <br />
- UNet3D is trained using all data like UNet2D <br />

## Visualiasion of Test Results
Using the test dataset, which is year 2022, below are show some results in binary classification and some from QGIS, binary classification is converted to Shapefile and visualised using QGIS and Google Maps as a Basemap. <br /> <br />

 - Red is the Ground Truth and Black box is the Prediction by UNet3D trained in All Samples (2006 to 2000), Green Points are the ignition points.

<img src="https://github.com/nikos230/WildFireSpread/blob/main/screenshots/Picture2.png">  <br /> <br />

- Binary Classification results

<img src="https://github.com/nikos230/WildFireSpread/blob/main/screenshots/Screenshot_2.jpg">


## Contact
For more info contact : nikolas619065@gmail.com <br /><br /><br />

## Dataset Variables, Spatial resolution and Sources

| Variables (Dynamic)                          | Spatial Resolution | Temportal Resolution | Source |  
|----------------------------------------------|:------------------:|:----------:|:-----------:|
|Max Temperature                               | 9km                |     1day    | ERA5-Land  |  
|Max Wind Speed                                | 9km                |     1day    | ERA5-Land  |
|Max Wind Direction                            | 9km                |     1day    | ERA5-Land  |
|Max Surface Pressure                          | 9km                |     1day    | ERA5-Land  |
|Min Relative Humidity                         | 9km                |     1day    | ERA5-Land  |
|Total Precipitation                           | 9km                |     1day    | ERA5-Land  |
|Mean Surface Solar Radiation Downwards        | 9km                |     1day    | ERA5-Land  |
|Day Land Surface Temperature                  | 1km                |     1day    | MODIS      |
|Night Land Surface Temperature                | 1km                |     1day    | MODIS      |
|Normalized Difference Vegetation Index (NDVI) | 500m               |     16days  | MODIS      |
|Leaf Area Index (LAI)                         | 500m               |     16days  | MODIS      |
|Soil moisture                                 | 5km                |     5days   | EDO        |
|Burned Areas                                  | 1km                |     1day    | EFFIS      |
|Ignition Points                               | 1km                |     1day    | MODIS      |

| Variables (Static)                           | Spatial Resolution | Source     |
|----------------------------------------------|:------------------:|:----------:|
|Slope                                         |30m                 | COP-DEM     
|Aspect                                        |30m                 | COP-DEM
|Curvature                                     |30m                 | COP-DEM
|Population                                    |1km                 | Worldpop
|Fraction of agriculture                       |300m                | Copernicus CCS
|Fraction of forest                            |300m                | Copernicus CCS
|Fraction of grassland                         |300m                | Copernicus CCS
|Fraction of settlements                       |300m                | Copernicus CCS
|Fraction of shrubland                         |300m                | Copernicus CCS
|Fraction of sparse vegetation                 |300m                | Copernicus CCS
|Fraction of water bodies                      |300m                | Copernicus CCS
|Fraction of wetland                           |300m                | Copernicus CCS
|Roads distance                                |1km                 | Worldpop

The above variables are in 64km x 64km x 7days resoltuion, for every fire event there is a 64x64km patch around the fire with a random offset so the burned area is not always in the middle of the patch, and for every sample there are 7 days in total, 5 before the fire started and 2 days after.


