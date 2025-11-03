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
**Project:** OpERA  
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
## 🏗️ Model Architecture

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
## ⚛️ The Physics – Beer-Lambert Law
Light attenuation in water follows the exponential decay relationship:
```math
R(z) = R_{\infty} + (R_0 - R_{\infty}) \times e^{-kz}
```
| Symbol   | Meaning                 | Typical Range |
| :------- | :---------------------- | :------------ |
| **R(z)** | Reflectance at depth z  | 0.05–0.15     |
| **R₀**   | Shallow reflectance     | ~0.15         |
| **R∞**   | Deep reflectance        | ~0.05         |
| **k**    | Attenuation coefficient | 0.05–0.25 m⁻¹ |
| **z**    | Depth (m)               | 0–30 m        |

## 🧮 Single Training Step (Simplified)
```python
# 1. Get batch of data
reflectance, true_depth = get_batch(batch_size=64)

# 2. Forward pass through ResNet
predicted_depth = resnet(reflectance)

# 3. Apply constraints
predicted_depth = clip(predicted_depth, 0, 30)

# 4. Data loss
loss_data = MSE(true_depth, predicted_depth)

# 5. Simulate physics (Beer–Lambert)
R_sim = Rinf + (R0 - Rinf) * exp(-k * predicted_depth)

# 6. Physics loss
loss_physics = MSE(reflectance, R_sim)

# 7. Total loss
total_loss = loss_data + λ * loss_physics

# 8. Backpropagation
gradients = compute_gradients(total_loss)
gradients = clip_gradients(gradients, max_norm=1.0)

# 9. Update parameters
update(model_weights, k, R0, Rinf)

# 10. Update metrics
log(RMSE, MAE, loss_data, loss_physics)
```
## 📉 Loss Functions
| Type                   | Equation                                       | Purpose                           |
| ---------------------- | ---------------------------------------------- | --------------------------------- |
| **Data Loss (MSE)**    | ( \frac{1}{n}\sum (z_{true} - z_{pred})^2 )    | Enforces accuracy to ground truth |
| **Physics Loss (MSE)** | ( \frac{1}{n}\sum (R_{obs} - R_{sim})^2 )      | Enforces Beer-Lambert consistency |
| **Total Loss**         | ( L_{total} = L_{data} + \lambda L_{physics} ) | Balances realism and accuracy     |
