![Atl text](https://github.com/nikos230/WildFireSpread/blob/main/logos/fire.png)<br /> <br /> 
(work in progress... as of Feb 2025)

![Alt text](https://github.com/nikos230/WildFireSpread/blob/main/screenshots/dynamic_variables.gif)

## About The Project
A Machine Learning ready Dataset and Model to predict final burned area. <br /> <br /> 
This project is part of my Thesis and makes use of [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) Dataset and a UNET model to predict final burned area from 28 remote sensing variables, Models such as UNet2D and UNet3D have been tested with spatial and temporal data and results are presented below.<br /><br /> 

### Dataset
---
Training and Testing of the Models have been done with [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) dataset. Training data (and validation) are 28 variables in 64x64km paches and a Spatial Resolution of 1 pixel = 1km x 1km in netCDF format. The mesogeos DataCube has 1km x 1km x 1 day Spatial and Temporal resolution. <br /> <br />
The Samples from the DataCube have 64km x 64km x 7days resoltuion, for every fire event there is a 64 x 64km patch around the fire with a random offset so the burned area is not always in the middle of the patch, and for every sample there are 7 days in total, 5 before the fire started and 2 days after. Samples are from the mesogeos region and includes 9500 samples from 31 countries and from years 2006 to 2022. <br /><br />


### Machine Learning Models
---
This project makes use of UNet2D and UNet3D models, the main difference in UNet3D is the 3D convolution which takes into account the temporal information of the samples as the 3D convolution can get info from 3 or more days at once, the 2D convolution is good for spartial feature extraction but not for temporal feature extraction. <br />
In Feature work the Dataset will be tested on a Vision Transformer (ViT) and results will be published. <br /><br />


### Models Evaluation and Metrics (latest results December 2024)
---
The main evaluation metric is the Dice Coefficient, presented below, which shows how much 2 shapes are similar, in this Segmatation Task along with Intersection Over Union (IoU) are the most important metrics. Using Samples with 64km x 64km x 7days resolution and training years from 2006 to 2020, validation 2021 (hyperparameters tuning) and 2022 for test the results are shown below. <br /> <br />
A baseline UNet2D is trained on only the fire day and then with all samples meaning using all 7 days each sample has. 


| Complexity     | UNet2D<br /> Baseline (1 day) | UNet2D<br /> All Samples  | UNet2D <br /> > 800ha | UNet2D <br /> > 1500ha | UNet3D <br /> All Samples
|----------------|:-----------------------------:|:-------------------------:|:---------------------:|:----------------------:|:-------------------------:
| [64, 128]      | 48.8                          | 47.7                      | 47.3                  | 46.7                   | **49.2**                          
| [64, 128, 256] | 48.8                          | 47.6                      | -                     | -                      | -

<br />Baseline model is using tesnor shape (channels, height, width) = (27, 64, 74) <br />
All Samples models are using tensor shape (channels, height, width) = (117, 64, 64) <br />
All Samples UNet3D model is using tensor shape (time, channels, height, width) = (7 , 27, 64, 64) <br /> <br />

### Visualiasion of Test Results
---
Using the test dataset, which is year 2021 and 2022, below are show some results in binary classification and some from QGIS, binary classification is converted to Shapefile and visualised using QGIS and Google Maps as a Basemap. <br /> <br />

 - Red is the Ground Truth and Black box is the Prediction by UNet3D trained in All Samples (2006 to 2000), Green Points are the ignition points.

<img src="https://github.com/nikos230/WildFireSpread/blob/main/screenshots/Picture2.png">  <br /> <br />

- Binary Classification results

<img src="https://github.com/nikos230/WildFireSpread/blob/main/screenshots/Screenshot_2.jpg">


## Contact
For more info contact in email : nikolas619065@gmail.com <br /><br /><br />

### Dataset Variables, Spatial resolution and Sources

| Variables (Dynamic, 1 day resolution)        | Spatial Resolution | Source     |  
|----------------------------------------------|:------------------:|:----------:|
|Max Temperature                               | 9km                | ERA5-Land  |
|Max Wind Speed                                | 9km                | ERA5-Land  |
|Max Wind Direction                            | 9km                | ERA5-Land  |
|Max Surface Pressure                          | 9km                | ERA5-Land  |
|Min Relative Humidity                         | 9km                | ERA5-Land  |
|Total Precipitation                           | 9km                | ERA5-Land  |
|Mean Surface Solar Radiation Downwards        | 9km                | ERA5-Land  |
|Day Land Surface Temperature                  | 1km                | MODIS      |
|Night Land Surface Temperature                | 1km                | MODIS      |
|Normalized Difference Vegetation Index (NDVI) | 500m               | MODIS      |
|Leaf Area Index (LAI)                         | 500m               | MODIS      |
|Soil moisture                                 | 5km                | EDO        |
|Burned Areas                                  | 1km                | EFFIS      |
|Ignition Points                               | 1km                | MODIS      |

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


