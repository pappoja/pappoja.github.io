# Enhancing Satellite Image Classification with Non-Image Data

## Introduction
In the evolving landscape of machine learning, models have progressed from handling simple tabular data to processing more complex data types. Feed-forward neural networks (FFNNs) are well-suited for tabular data, convolutional neural networks (CNNs) for images, and recurrent neural networks (RNNs) or transformers for language and other time series data. More recently, researchers have become interested in combining these foundational architectures into multi-modal hybrid models.  
  
In this paper, we investigate whether image classification can be enhanced by integrating structured non-image data into model architectures. Our focus is on satellite image classification, where auxiliary geographic and environmental variables may provide signal beyond what is available in raw pixels. These features can enrich the model’s understanding and lead to improved predictive performance.  
  
We explore this idea using the EuroSAT dataset, which contains 27,000 satellite images categorized into ten land use and land cover classes. Building on benchmark CNN-based models established by Helber et al., we incorporate non-image data into computer vision architectures to create a multi-modal learning framework. The non-image data is collected from Google Earth Engine, which provides various geospatial measurements from satellite sensor technologies.  
  
This paper proposes a general framework for the effective fusion of image and non-image modalities, while also presenting a concrete implementation in the context of Earth observation. Specifically, we investigate how satellite imagery and satellite-derived measurements can be integrated within a unified model architecture to improve land use and land cover classification. By addressing both broader architectural considerations and a targeted application, we aim to contribute insights that are relevant not only for remote sensing tasks but also for the design of multi-modal machine learning systems in general.

## Related Works
Early breakthroughs in computer vision came with convolutional neural networks (CNNs), introduced by LeCun et al. for digit recognition, and later extended to large-scale image classification tasks with AlexNet [1]. Residual networks (ResNets) introduced skip connections, which allow gradients to flow more effectively through the network and preserve features learned in earlier layers [2]. This enabled the successful training of much deeper architectures. The 50-layer variant, ResNet-50, has become widely used in visual recognition tasks due to its strong performance and scalability.  
  
In the field of earth observation machine learning (EO-ML), CNNs have been employed to process satellite imagery and predict variables related to land use, agriculture, and human development. The EuroSAT dataset is one prominent example, enabling classification of land cover types from RGB and multispectral satellite images [3]. Other studies have extended this approach to infer socioeconomic indicators. Yeh et al. used CNNs on satellite imagery to estimate economic development and poverty [4]. Beyond static images, EO-ML has increasingly embraced spatio-temporal modeling. CNNs can be used to extract spatial features from satellite images and feed these outputs into recurrent architectures such as LSTMs to capture changes over time [5].  
  
Outside of EO-ML, computer vision has seen a growing interest in combining image data with non-image modalities, such as tabular or textual features, to enhance prediction. For example, CLIP, a model that jointly trains on image-text pairs to learn visual representations, laid the foundational groundwork for vision-language models capable of zero-shot transfer across downstream tasks [6].  
  
Our work builds on advances in multi-modal learning, focusing specifically on earth observation data. We explore how structured non-image data can be fused into CNN-based architectures to improve land use classification for the EuroSAT dataset.


## Methods
### Data Collection and Preprocessing
#### Image Data
Our primary dataset is EuroSAT, a publicly available collection of 27,000 images taken by the Sentinel-2A satellite [3]. Each is 64×64 pixels, with a ground resolution of up to 10 meters per pixel. Thus, each has a resolution of 640 meters in length and height, or around 0.16 square miles. The images are labeled into one of ten land use and land cover classes: Forest, Annual Crop, Permanent Crop, Pasture, Herbaceous Vegetation, Residential, Industrial, Highway, River, and Sea Lake (see Figure 1). Although the EuroSAT paper does not specify exact acquisition dates of the images they use, the earliest date mentioned is August 2015 (two months after the launch of Sentinel-2A) and the latest is March 2017.  
<figure>
  <img src="images/figure1_eurosat.png" alt="Sample EuroSAT Images" width="1000"/>
  <figcaption><em>Figure 1: Samples of the 10 land cover and land usage labels in EuroSAT. [3]</em></figcaption>
</figure>  

#### Non-Image Data
To supplement the images with non-image data, we first extracted the latitude and longitude coordinates embedded in the metadata of the EuroSAT multispectral .tif files. Each image’s geographic center was then mapped to a country based on shapefile boundaries.  
  
We then queried environmental and geospatial variables from Google Earth Engine (GEE). Specifically, we retrieved: (1) soil moisture from NASA/SMAP/SPL3SMP_E/006 (9,000 meters/pixel), (2) surface elevation from USGS/SRTMGL1_003 (30 meters/pixel), (3) terrain slope from elevation models, (4) nighttime light intensity from NOAA/VIIRS/DNB (464 meters/pixel), (5) vegetation density (NDVI) from MODIS/061/MOD13A2 (1,000 meters/pixel), (6) surface temperature from MODIS/061/MOD11A2 (1,000 meters/pixel), (7) surface humidity derived from temperature and dewpoint values in ECMWF/ERA5/DAILY (11,132 meters/pixel), and (8) population from WorldPop/GP/100m/pop (93 meters/pixel) [7]. We record measurements at the resolution of the one pixel which contains the coordinate associated with the corresponding EuroSAT image in order to ensure spatial alignment between resolutions. The latitude and longitude were also included among the non-image features.  
  
These geospatial measurements are derived from physical modeling and satellite-based sensor data. Although they can be visualized spatially, their coarse resolution relative to the fine-grained EuroSAT image patches makes it more appropriate to treat them as tabular features. Unlike image data, they do not encode texture or visual patterns but represent aggregated environmental attributes over larger geographic grids.  
  
To maintain temporal alignment across input features, all time-varying non-image variables were averaged over the full year of 2016. This decision aligns with the timeframe of the EuroSAT image acquisitions, which, as mentioned above, likely range from August 2015 to March 2017. Helber et al. sampled satellite images from across the year to capture seasonal variance, further motivating our decision to limit our non-image data to one calendar year.  
  
### EuroSAT Benchmarks
In the original EuroSAT paper, Helber et al. benchmarked several models across different band combinations, weight initializations, and data splits. The models tested included a bag-of-visual-words (BoVW) classifier using SIFT features, a simple 2-layer CNN, GoogleNet, and ResNet-50. For weight initialization, Helber et al. experimented with training models from scratch and fine-tuning from models pre-trained on ILSVRC-2012. They experimented with multiple data splits, with an 80-20 train-test ratio having the best performance. Images were classified using different spectral bands, including color-infrared (CI), short-wave infrared (SWIR), and RGB, the latter of which yielded the highest performance. This benchmark setup informs our own modeling choices, providing a strong and well-understood baseline for evaluating the impact of additional non-image features. See the results from Helber et al. in Table I.  
<figure>
  <img src="images/table1_eurosat.png" alt="Sample EuroSAT Images" width="1000"/>
  <figcaption><em>Table I: Classification accuracy across different train-test splits from Helber et al. [3]</em></figcaption>
</figure>

