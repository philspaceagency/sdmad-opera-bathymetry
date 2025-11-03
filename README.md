# sdmad-opera-bathymetry

# 🌊 Physics-Informed Neural Network (PINN)
**Complete Training Explanation for Bathymetry Estimation**

---

## 🎯 What is This Model?

A **Physics-Informed Neural Network (PINN)** combines:

- 🤖 **Neural Network** — Learns patterns from data (Sentinel-2 reflectance → depth)  
- ⚛️ **Physics Laws** — Beer-Lambert Law (light attenuation in water)  
- 📊 **Best of Both** — Accurate + Realistic predictions

> **Key Insight:** Traditional models only learn from data. PINN also learns from physics, making predictions more accurate and physically realistic.

---

## 🏗️ Model Architecture

### Tabs
- **ResNet Base**
- **PINN Wrapper**
- **Complete Flow**

### 1. Base Model: ResNet

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
