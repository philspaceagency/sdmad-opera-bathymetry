# sdmad-opera-bathymetry

# Getting started 
------------------
## The repository contais five (5) jupyter notebooks
- Module 01: Downloading Sentinel2 as TFRecord
- Module 02: Extracting Reflectance Values using the Bathymetry file
- Module 03: Preparation of the Bathymetry file with corresponding reflectance values
- Module 04: Training Physics-Informed Neural Network using the input dataset
- Module 05: Inference on unseen data (other areas)


## Prerequisites
-----------------
Before starting, ensure you have: 
- **Python 3.12** installed 
- **Tensorflow** installed
- **Google Earth Engine** Account, needed to download large scale satellite images (Sentinel2)

# 🌊 OpERA: Estimating Bathymetry of Shallow Water in the Philippines using Sentinel-2 and Physics-Informed Neural Networks (PINN)

**Author:** Christian Candido  
**Organization:** Philippine Space Agency – Space Data Mobilization and Applications Division (SDMAD)  
**Project:** OpERA (Optical Estimation of Reef and Aquatic Depth)  
**Keywords:** Bathymetry • Sentinel-2 • Physics-Informed Neural Network • Beer-Lambert Law • ResNet • Remote Sensing  

---

## 🎯 Project Overview

This project aims to **estimate shallow water bathymetry** (0–30 m depth) across coastal areas in the Philippines using **Sentinel-2 multispectral imagery** and a **Physics-Informed Neural Network (PINN)**.

Unlike traditional data-driven models, the **PINN** integrates both **neural learning** and **physical constraints** derived from the **Beer-Lambert law of light attenuation in water**, resulting in bathymetric predictions that are both *accurate* and *physically consistent*.

---

## 🧭 Workflow

```sqe
graph TD
    A[Sentinel-2 Reflectance Data] --> B[Training Dataset Preparation]
    B --> C[ResNet Base Model]
    C --> D[PINN Wrapper: Physics Constraints]
    D --> E[Training: Data + Physics Loss]
    E --> F[Optimized Model Parameters]
    F --> G[Depth Prediction Map]
    G --> H[Evaluation Metrics (RMSE, R²)]
```
🏗️ Model Architecture

Input: 8 Sentinel-2 Bands [B1, B2, B3, B4, B8, B8A, B11, B12]
Output: Predicted Depth (single value per pixel)
```scss
🔹 Base: ResNet Structure
Input → Dense(64) + BatchNorm + ReLU
     ↓
Residual Block 1 → Dense → BN → Dropout → Dense → Add(residual) → ReLU
     ↓
Residual Block 2 → 128 filters
Residual Block 3 → 256 filters
     ↓
Dense(64) + Dropout(0.3)
     ↓
Output: Depth
```
