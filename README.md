# sdmad-opera-bathymetry

## Getting started 
------------------
### The repository contais five (5) jupyter notebooks
- Module 01: Downloading Sentinel2 as TFRecord
- Module 02: Extracting Reflectance Values using the Bathymetry file
- Module 03: Preparation of the Bathymetry file with corresponding reflectance values
- Module 04: Training Physics-Informed Neural Network using the input dataset
- Module 05: Inference on unseen data (other areas)


### Prerequisites
-----------------
Before starting, ensure you have: 
- **Python 3.12** installed 
- **Tensorflow** installed
- **Google Earth Engine** Account, needed to download large scale satellite images (Sentinel2)

### Model Training 
------------------
### 🌊 Physics-Informed Neural Network (PINN)
**Complete Training Explanation for Bathymetry Estimation**

#### 🎯 What is This Model?

A **Physics-Informed Neural Network (PINN)** combines:

- 🤖 **Neural Network** — Learns patterns from data (Sentinel-2 reflectance → depth)  
- ⚛️ **Physics Laws** — Beer-Lambert Law (light attenuation in water)  
- 📊 **Best of Both** — Accurate + Realistic predictions

> **Key Insight:** Traditional models only learn from data. PINN also learns from physics, making predictions more accurate and physically realistic.
#### Base Model
Input: 8 Sentinel-2 bands [B1, B2, B3, B4, B8, B8A, B11, B12]
↓
Dense(64) + BatchNorm + ReLU
↓
[Residual Block 1] 64 filters
Dense → BatchNorm → Dropout → Dense → Add(residual) → ReLU
↓
[Residual Block 2] 128 filters
Dense → BatchNorm → Dropout → Dense → Add(residual) → ReLU
↓
[Residual Block 3] 256 filters
Dense → BatchNorm → Dropout → Dense → Add(residual) → ReLU
↓
Dense(64) + Dropout(0.3)
↓
Output: Predicted Depth (1 value)

#### PINN Wrapper

Wraps the base model and adds physics constraints.

```python
class PINNWrapper:
    base_model       # ResNet that predicts depth
    k                # Attenuation coefficient (learnable!)
    R0               # Shallow water reflectance (learnable!)
    Rinf             # Deep water reflectance (learnable!)
    lambda_phy       # Physics loss weight (adaptive!)
```

```yaml
Would you like me to:
- Save the `README.md` into your local repo (I can provide a small script to write it locally), or  
- Create a ready-to-commit file content and show the `git` commands to add/commit/push it?

Which one do you want next?
```
