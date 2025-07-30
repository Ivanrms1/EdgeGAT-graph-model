
# EdgeGAT: Hybrid Graph Attention Network for Semantic Segmentation of Plant Components in 3D Point Clouds

This repository contains the official implementation of **EdgeGAT**, a hybrid graph neural network architecture designed for semantic segmentation of plant components (soil, stem, and leaves) from 3D point cloud data.

---

##  Overview

The **EdgeGAT** model integrates **EdgeConv** layers for effective local feature extraction and **Graph Attention Networks (GAT)** for global context modeling and final point-wise classification. It leverages PCA-based feature augmentation and careful data augmentation strategies to improve generalization, especially suited for agricultural applications involving plant phenotyping.

---

## Requirements and Installation

First, create a Python virtual environment and install dependencies:

# Dataset Preparation
Datasets used in this work are derived from:

Ao et al. dataset

Pheno4D dataset

### Data augmentation

Due files in Ao dataset was of 30-40 PC plants and -70 for pheno4D dataset we applied data augmentation like -jittering - scalation - rotation and translation in order to enchance data variability. This lead to have 4 times data for each dataset. 

## Ablation study

We performed a small ablation study in order to view impact of PCA features extraction before introducing to model, we perfomed this study with well-know graph models, like Simple GCN, GAT, DGCNN and UnetGCN variations. This files was perfomed by the next acronyms: 
XYZ Raw 3D coordinates
N Surface normals
C Curvature
L Linearity
P Planarity
S Scattering
O Omnivariance
A Anisotropy
E Eigenentropy

This features can be extracted by argparde on python training files, it can introduce "all" for extract all of this PCA features listed before, after the feature enchance factor, we normalice between all other features. 

Also there was introduce a feature for Laplacian eigenvalors extraction that can be added as a feature, for this u must need change to "True" in  train.py file. 
There is also 2 differents train files, train_k.py is prepared to use k-cross validation over X folds, by default i use to 5 folds, but u can change it by calling as argument, final argument example can be: python3 train_k.py EdgeGAT all 5, being this last number the number of folds used in k cross validation, train.py don't manage k cross, so u can't omit on training. 

## Model selection, 
The original model of my studie was with name of EdgeGAT, but i include other models that i used before, like GCN, GAT, GCNUnet, it can be called like "GAT", "Unet", "Unet2", "GCN", and other mixture of models.

### Standar hyperparameters
I managed to use similar hyperparameters, like 16 k nearest neighbors, batch size of 8, learning rates of 0.0001 and 100 epochs.

## EdgeGAT model

This is the core architecture, similar to other models it is based on EdgeConv for aditional feature enhancing, can be viewed as a local feature extractor between nodes and neighborhods and Graph attention layers for final clasification. The complete architecture can be revised above, it consist on a 2-layers of EdgeConv after PCA feature extraction, and finishes with 2 Gat FC layers in node clasification.
The result tables can be viewed at https://arxiv.org/html/2507.00182v1. 

