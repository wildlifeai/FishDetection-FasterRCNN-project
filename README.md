# Spyfish Aotearoa - FishDetection with FasterRCNN

This repository contains scripts and resources to train machine learning algorithms to identify fish for Spyfish Aotearoa using data labelled by citizen scientists.

Adi Gabay and Ohad Tayler carried out this work, as part of their internship in Wildlife.ai and their studies in the Hebrew University.

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Overview
To train the fish identification models we used still images of baited underwater videos (BUV). BUV is a simple, versatile underwater survey method that researchers use to count scavenger fish inside and outside marine reserves. 

Citizen scientists labelled the images at [Spyfish Aotearoa Zooniverse project][Spyfish-link]. Using the [Koster data mangament tools][koster_data_man], we aggregated the labels of multiple citizen scientists and retrieved the data in a folder with the images and a folder of the labels in YOLO format.

We trained a 'FasterRCNN' using the pipeline below on the data to detect three fish species (blue cod, snapper and scarlet wrasse). We used methods like augmentations, style transfer to improve our model results.



ADD IMAGE OF THE FLOWCHART OF THE WORKING PIPELINE




## Requirements
* [Python 3.7+](https://www.python.org/)
* GPU specs?
* Trainig data (COCO format).

## How to use
1. In order to train the model you should run the train.py with the required parameters.
2. To test the model you should run the file test.py
3. In order to load the data and transform the data you should run the SpyFishAotearoaDatahandler - this should create a folder with the formatted data and the CSV files.
4. To get general information about the data you should run the file utils/eda_lib.py
5. To transfer the data style you need to use the utils/transfer_dataset
6. To plot the images with the labels you should run the file plot_boxes_on_images.py

## Citation

If you use this code or its models, please cite:

Gabay A, Taylor O, Paz E, Hyams G, Anton V (2022). Spyfish Aotearoa - FishDetection with FasterRCNN. https://github.com/wildlifeai/FishDetection-FasterRCNN-project


## Collaborations/questions

We are working to make our work available to anyone interested. Please feel free to [contact us][contact_info] if you have any questions.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/wildlifeai/FishDetection-FasterRCNN-project.svg?style=for-the-badge
[contributors-url]: https://https://github.com/wildlifeai/FishDetection-FasterRCNN-project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildlifeai/FishDetection-FasterRCNN-project.svg?style=for-the-badge
[forks-url]: https://github.com/wildlifeai/FishDetection-FasterRCNN-project/network/members
[stars-shield]: https://img.shields.io/github/stars/wildlifeai/FishDetection-FasterRCNN-project.svg?style=for-the-badge
[stars-url]: https://github.com/wildlifeai/FishDetection-FasterRCNN-project/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildlifeai/FishDetection-FasterRCNN-project.svg?style=for-the-badge
[issues-url]: https://github.com/wildlifeai/FishDetection-FasterRCNN-project/issues
[license-shield]: https://img.shields.io/github/license/wildlifeai/FishDetection-FasterRCNN-project.svg?style=for-the-badge
[license-url]: https://github.com/wildlifeai/FishDetection-FasterRCNN-project/blob/main/LICENSE.txt
[Spyfish-link]: https://www.zooniverse.org/projects/victorav/spyfish-aotearoa
[koster_data_man]: https://github.com/ocean-data-factory-sweden/koster_data_management
[contact_info]: contact@wildlife.ai

