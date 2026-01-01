# atrium_segmentation
# Left Atrium Segmentation from Cardiac MRI using U-Net

## Overview
This project implements an end-to-end deep learning pipeline for **left atrium segmentation** from **cardiac MRI scans** using a **U-Net architecture** in PyTorch and PyTorch Lightning.  
The workflow includes data loading from NIfTI files, preprocessing, augmentation, model training, validation, and inference on unseen subjects.

The project is inspired by medical image segmentation tasks such as those found in the **Medical Segmentation Decathlon (Task02: Heart)**.

## Methodology
1. Load 3D cardiac MRI volumes 
2. Slice volumes into 2D images
3. Normalize and standardize intensity values
4. Apply data augmentation
5. Train a U-Net model using Dice Loss
6. Evaluate segmentation performance
7. Perform inference and visualize predictions

## Preprocessing
- Center cropping (`32:-32`)
- Intensity normalization (zero mean, unit variance)
- Min-max standardization
- Slice-wise saving as `.npy`
- Train/validation split

## Data Augmentation
Implemented using **imgaug**:
- Affine transformations (scaling, rotation)
- Elastic deformation  
Augmentations are applied consistently to both images and masks.

## Model Architecture
A custom **U-Net** consisting of:
- Encoder: 4 DoubleConv blocks with max pooling
- Decoder: Upsampling with skip connections
- Final 1×1 convolution for binary segmentation

## Dataset
- **Input:** Cardiac MRI scans in NIfTI format  
- **Labels:** Binary segmentation masks of the left atrium  
- **Orientation:** RAS (Right–Anterior–Superior)

