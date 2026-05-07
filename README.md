# PhilSA OpERA Bathymetry

**Physics-Informed Neural Network for Sentinel-2 Bathymetry Estimation**

A deep learning pipeline for coastal water depth mapping using Sentinel-2 satellite imagery, combining ACOLITE atmospheric correction with a Transformer-based PINN architecture constrained by Beer-Lambert optical physics.

## Overview

This project estimates nearshore bathymetry (water depth) from Sentinel-2 multispectral imagery using a Physics-Informed Neural Network (PINN). The model incorporates the Beer-Lambert law as a physical constraint during training, improving generalization in optically shallow waters.

### Workflow

```
GPKG tiles
    └─► Sentinel-2 L1C collection (cloud-filtered)
            └─► ACOLITE DSF atmospheric correction + deglinting
                    └─► Xarray DataArray (band, y, x)
                            └─► PINN bathymetry estimation
                                    └─► Temporal mean bathymetry
```

## Model Architecture

The PINN uses a Transformer encoder backbone that treats each spectral band as an independent token, learning cross-band attention relationships:

- **Input**: 8 Sentinel-2 bands (B1, B2, B3, B4, B8, B8A, B11, B12) as remote sensing reflectance (Rrs)
- **Architecture**: 3-layer Transformer encoder with 4 attention heads, 64-dim embeddings
- **Physics constraint**: Beer-Lambert law `R(z) = R_inf + (R_0 − R_inf) · exp(−k · z)`
- **Output**: Water depth in meters (clipped to 0-30m range)

### Performance Metrics

| Metric | Value |
|--------|-------|
| RMSE | 0.67 m |
| MAE | 0.36 m |
| R² | 0.95 |
| Bias | 0.037 m |

## Project Structure

```
.
├── notebooks/
│   ├── bathymetry_model.py          # Core PINN model and inference code
│   ├── v1/                          # Version 1 notebooks
│   │   ├── Module01_DownloadS2TFRecord.ipynb
│   │   ├── Module02_DataExtraction.ipynb
│   │   ├── Module03_DataPreparation.ipynb
│   │   ├── Module04_ModelTraining_PINN.ipynb
│   │   └── Module05_ModelInference.ipynb
│   ├── Module01_Inference-ACOLITE.ipynb        # Full pipeline with ACOLITE DSF
│   ├── Module02_DataPreparation-ACOLITE.ipynb
│   └── Module03_ModelTrainingPINN-ACOLITE.ipynb
│
├── model/
│   ├── PINN/                        # Original PINN model files
│   │   ├── base_model.keras
│   │   ├── physics_params.npy
│   │   ├── training_history.csv
│   │   └── pinn_predictions.png
│   └── PINNv2/                      # Updated PINN model
│       ├── base_model.keras
│       ├── physics_params.json      # k, R0, Rinf coefficients
│       ├── pinn_weights.weights.h5
│       ├── training_history.csv
│       ├── metrics.json
│       └── pinn_predictions_evaluation.png
│
└── README.md
```

## Installation

```bash
pip install tensorflow scikit-learn xarray dask[array]
pip install pystac_client stackstac planetary_computer xee
pip install rasterio rio-cogeo netcdf4 geopandas
```

## Usage

### Loading the Model

```python
from bathymetry_model import load_bathymetry_model

# Load trained PINN model
infer = load_bathymetry_model(
    model_dir='/path/to/model',
    depth_min=0.0,
    depth_max=30.0,
    batch_size=2048
)
```

### Predicting Bathymetry

```python
import xarray as xr

# Predict from xarray DataArray (band, y, x)
depth = infer.predict_xarray(scene_da)

# Or predict from numpy array (n_pixels, 8)
depth = infer.predict_array(reflectance_array)
```

### Running the Full Pipeline

See `notebooks/Module01_Inference-ACOLITE.ipynb` for the complete workflow:

1. Load tile geometries from GeoPackage
2. Query Sentinel-2 L1C scenes via Google Earth Engine
3. Apply cloud/shadow masking
4. Run ACOLITE DSF atmospheric correction and deglinting
5. Estimate bathymetry per scene with PINN
6. Compute temporal mean bathymetry
7. Export as Cloud-Optimized GeoTIFF (COG)

## Input Requirements

| Parameter | Value |
|-----------|-------|
| **Bands** | B1, B2, B3, B4, B8, B8A, B11, B12 |
| **Format** | Surface reflectance (Rrs) in [0, 1] |
| **Source** | Sentinel-2 L1C, corrected with ACOLITE DSF |
| **Depth range** | 0 - 30 meters |

## Model Files

| File | Description |
|------|-------------|
| `physics_params.json` | Beer-Lambert parameters (k, R0, Rinf) |
| `pinn_weights.weights.h5` | Trained model weights |
| `scaler.pkl` | Optional sklearn StandardScaler |

## License

MIT License

## Contact

PhilSA OpERA Project - Philippine Space Agency
