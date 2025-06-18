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
  <img src="images/figure1_eurosat.png" alt="Sample EuroSAT Images" width="500"/>
  <figcaption>Figure 1: Samples of the 10 land cover and land usage labels in EuroSAT. [3]</figcaption>
</figure>
