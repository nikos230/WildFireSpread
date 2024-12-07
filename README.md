![Atl text](https://github.com/nikos230/WildFireSpread/blob/main/logos/logo3.png)

## About The Project
A Machine Learning ready Dataset and Model to predict final burned area. <br /> <br /> This project make use of [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) Dataset and a UNET model to predict final burned area from 28 remote sensing variables out of which 14 are static and 14 dynamic (wind speed, wind direction, ndvi etc.). Models such as UNet2D and UNet3D have been tested with spatial and temporal data.

### Dataset
Training and Testing of the Models have been done with [mesogeos](https://github.com/Orion-AI-Lab/mesogeos) dataset. Training data (and validation) are 28 variables in 64x64km paches in netCDF format.


| Variable                                     | Spatial Resolution | Source    |
|----------------------------------------------|:------------------:|----------:|
|Max Temperature                               | 9km                | ERA5-Land |
|Max Wind Speed                                | 9km                | ERA5-Land |
|Max Wind Direction                            | 9km                | ERA5-Land |
|Max Surface Pressure                          | 9km                | ERA5-Land |
|Min Relative Humidity                         | 9km                | ERA5-Land |
|Total Precipitation                           | 9km                | ERA5-Land |
|Mean Surface Solar Radiation Downwards        | 9km                | ERA5-Land |
|Day Land Surface Temperature                  | 1km                | MODIS     |
|Night Land Surface Temperature                | 1km                | MODIS     |
|Normalized Difference Vegetation Index (NDVI) | 500m               | MODIS     |
|Leaf Area Index (LAI)                         | 500m               | MODIS     |
|Soil moisture                                 | 5km                | EDO       |
|Burned Areas                                  | 1km                | EFFIS     |
|Ignition Points                               | 1km                | MODIS     |
|                                              |               |       |
