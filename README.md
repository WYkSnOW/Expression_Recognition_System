## Facial Expression Recognition System Using Machine Learning

### By:
- Waiyuk Kwong
- Zhihui Chen
- Tyler Lin
- Blane R. York
- Carter D. Robinson

## Setup Instructions

- This repository contains large model fils that require Git Large File Storage LFS for handling. 
- Pleas sure you have Git LFS installed before cloning the repository. You can download and install Git LFS from https://git-lfs.com/

## Preparing Model Files

#### In the ml_model folder, you’ll find two compressed model files. Upzip each model file:

1. svm.zip
2. rf.zip

#### Important: Ensure that unzippedl model files remain in the ml_model folder.

## Directory and Files Description

- /archive/: directory for test and train data
- /archive/test/: test data images
- /archive/train/: train data images

- /data_processing_method/: directory for preprocessing data
- /data_processing_method/cnn_image_processing_pipeline.py: process image for cnn
- /data_processing_method/data_augmentation.py: augment data
- /data_processing_method/face_mesh_module.py: face mesh detector
- /data_processing_method/image_normalization.py: normalize image
- /data_processing_method/image_processing_pipeline.py:  image processing

- /evaluation/: code to evaluate models
- /evaluation/evaluation_cnn.py: evaluate cnn model
- /evaluation/evaluation_rf.py: evaluate rf model
- /evaluation/evaluation_svm.py: evaluate svm model

- /helper/: helper functions

- /ml_model/: machine learning models

- /streamlit/: images of chart analysis to show on streamlit

- /train/: code to train models
- /train/train_cnn_model.py: train cnn model
- /train/train_rf_model.py: train rf model
- /train/train_svm_model.py: train svm model

- /views/: page navigation for streamlit

