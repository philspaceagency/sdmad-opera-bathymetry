"""
bathymetry_model.py
====================
Self-contained PINN (Physics-Informed Neural Network) inference module
for Sentinel-2 bathymetry estimation.

Architecture
------------
  Transformer backbone  (create_transformer_model)
      └─► PINNWrapper  (Beer-Lambert physics constraint)

Public API
----------
  load_bathymetry_model(model_dir, depth_max=30.0)
      → BathymetryInference

  BathymetryInference.predict_xarray(da_scene)
      → xr.DataArray (y, x)  depth in metres

  BathymetryInference.predict_array(reflectance_array)
      → np.ndarray (n_pixels,)  depth in metres

Expected file layout in model_dir
----------------------------------
  model_dir/
      physics_params.json      # k, R0, Rinf lists
      pinn_weights.weights.h5  # all trainable weights
      scaler.pkl               # sklearn StandardScaler (optional)
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Lazy TensorFlow import — avoids import-time crash if TF is not installed
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow as tf


# ============================================================================
# 1.  Model architecture
# ============================================================================

def create_transformer_model(
    input_shape: tuple = (8,),
    n_blocks: int = 3,
    embed_dim: int = 64,
    n_heads: int = 4,
    ff_dim: int = 128,
) -> Model:
    """
    Transformer-based spectral encoder for bathymetry depth prediction.

    Each spectral band is treated as an independent token, so the model
    learns cross-band attention (e.g. how B2 relates to B8).

    Args:
        input_shape : (n_bands,)  — default (8,) for B1,B2,B3,B4,B8,B8A,B11,B12
        n_blocks    : Number of Transformer encoder blocks.
        embed_dim   : Token embedding dimension (must be divisible by n_heads).
        n_heads     : Number of self-attention heads.
        ff_dim      : Hidden size of the feed-forward sub-layer.

    Returns:
        Compiled Keras Model.
    """
    assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

    n_bands = input_shape[0]
    inputs  = layers.Input(shape=input_shape)                     # (B, 8)

    # ── 1. Per-band embedding ─────────────────────────────────────────────
    x = layers.Reshape((n_bands, 1))(inputs)                      # (B, 8, 1)
    x = layers.Dense(embed_dim)(x)                                 # (B, 8, D)

    # ── 2. Learnable positional encoding ─────────────────────────────────
    positions    = tf.range(start=0, limit=n_bands, delta=1)
    pos_embed    = layers.Embedding(input_dim=n_bands, output_dim=embed_dim)(positions)
    x            = x + pos_embed                                   # (B, 8, D)

    # ── 3. Transformer encoder blocks ────────────────────────────────────
    for _ in range(n_blocks):
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embed_dim // n_heads, dropout=0.1
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn]))

        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(embed_dim)(ff)
        x  = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))

    # ── 4. Aggregate tokens → scalar depth ───────────────────────────────
    x       = layers.GlobalAveragePooling1D()(x)                   # (B, D)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)              # (B, 1)

    return Model(inputs=inputs, outputs=outputs, name="Transformer_Bathymetry")


# ============================================================================
# 2.  PINN wrapper
# ============================================================================

class PINNWrapper(Model):
    """
    Physics-Informed Neural Network for bathymetry estimation.

    Wraps any base regression model and adds a Beer-Lambert law physics
    constraint during training (ignored at inference time).

    Physical model:  R(z) = R_inf + (R_0 − R_inf) · exp(−k · z)

    Parameters
    ----------
    base_model      : Keras Model — the spectral encoder.
    n_bands         : Number of input spectral bands.
    lambda_phy      : Initial physics loss weight.
    depth_min/max   : Hard output clipping range (metres).
    adaptive_lambda : Whether to decay lambda during training.
    init_k          : Initial attenuation coefficients (n_bands,).
    init_R0         : Initial shallow-water reflectances (n_bands,).
    init_Rinf       : Initial deep-water reflectances (n_bands,).
    """

    def __init__(
        self,
        base_model: Model,
        n_bands: int = 8,
        lambda_phy: float = 0.1,
        depth_min: float = 0.0,
        depth_max: float = 30.0,
        adaptive_lambda: bool = False,
        init_k:    Optional[np.ndarray] = None,
        init_R0:   Optional[np.ndarray] = None,
        init_Rinf: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.base_model          = base_model
        self.n_bands             = n_bands
        self.depth_min           = depth_min
        self.depth_max           = depth_max
        self.adaptive_lambda     = adaptive_lambda
        self.lambda_phy_initial  = lambda_phy

        # ── Default physical initialisations ────────────────────────────
        if init_k    is None: init_k    = np.full(n_bands, 0.10, dtype=np.float32)
        if init_R0   is None: init_R0   = np.full(n_bands, 0.15, dtype=np.float32)
        if init_Rinf is None: init_Rinf = np.full(n_bands, 0.05, dtype=np.float32)

        init_k    = np.asarray(init_k,    dtype=np.float32)
        init_R0   = np.asarray(init_R0,   dtype=np.float32)
        init_Rinf = np.asarray(init_Rinf, dtype=np.float32)

        # ── Unconstrained raw variables (constraints applied via properties)
        # k     = softplus(raw_k)          → k > 0
        # Rinf  = sigmoid(raw_Rinf)        → 0 < Rinf < 1
        # R0    = Rinf + softplus(raw_δ)   → R0 > Rinf
        self._raw_k = tf.Variable(
            tf.math.log(np.exp(init_k) + 1e-7) - tf.math.log(1.0 + 1e-7),
            trainable=True, name="raw_k", dtype=tf.float32,
        )
        init_Rinf_clipped = np.clip(init_Rinf, 1e-6, 1.0 - 1e-6)
        self._raw_Rinf = tf.Variable(
            tf.math.log(init_Rinf_clipped / (1.0 - init_Rinf_clipped)),
            trainable=True, name="raw_Rinf", dtype=tf.float32,
        )
        init_delta = np.maximum(init_R0 - init_Rinf, 0.01)
        self._raw_delta = tf.Variable(
            tf.math.log(np.exp(init_delta) + 1e-7) - tf.math.log(1.0 + 1e-7),
            trainable=True, name="raw_delta", dtype=tf.float32,
        )
        self.lambda_phy = tf.Variable(
            lambda_phy, trainable=False, dtype=tf.float32
        )

        # ── Keras metrics ────────────────────────────────────────────────
        self.loss_fn             = tf.keras.losses.MeanSquaredError()
        self.rmse_metric         = tf.keras.metrics.RootMeanSquaredError(name="rmse")
        self.mae_metric          = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.data_loss_tracker   = tf.keras.metrics.Mean(name="loss_data")
        self.phy_loss_tracker    = tf.keras.metrics.Mean(name="loss_phy")
        self.total_loss_tracker  = tf.keras.metrics.Mean(name="loss")

    # ── Constrained physics properties ───────────────────────────────────

    @property
    def k(self) -> tf.Tensor:
        """Attenuation coefficients (k > 0)."""
        return tf.nn.softplus(self._raw_k)

    @property
    def Rinf(self) -> tf.Tensor:
        """Deep-water reflectance (0 < Rinf < 1)."""
        return tf.nn.sigmoid(self._raw_Rinf)

    @property
    def R0(self) -> tf.Tensor:
        """Shallow-water reflectance (R0 > Rinf)."""
        return self.Rinf + tf.nn.softplus(self._raw_delta)

    # ── Forward pass ─────────────────────────────────────────────────────

    def call(self, inputs, training: bool = False):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        return self.base_model(x, training=False)

    # ── Utilities ────────────────────────────────────────────────────────

    def apply_depth_constraint(self, depth_pred: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(depth_pred, self.depth_min, self.depth_max)

    def beer_lambert_law(self, depth: tf.Tensor) -> tf.Tensor:
        k    = tf.reshape(self.k,    [1, -1])
        R0   = tf.reshape(self.R0,   [1, -1])
        Rinf = tf.reshape(self.Rinf, [1, -1])
        z    = tf.expand_dims(depth, -1)
        return Rinf + (R0 - Rinf) * tf.exp(-k * z + 1e-7)

    def compute_physics_loss(
        self, reflectance_obs: tf.Tensor, depth_pred: tf.Tensor
    ) -> tf.Tensor:
        return self.loss_fn(reflectance_obs, self.beer_lambert_law(depth_pred))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, self.rmse_metric, self.mae_metric,
            self.data_loss_tracker,  self.phy_loss_tracker,
        ]

    # ── Training / evaluation steps ──────────────────────────────────────

    def train_step(self, data):
        (x, reflectance_obs), y_true = data
        with tf.GradientTape() as tape:
            y_pred     = self.apply_depth_constraint(self.base_model(x, training=True))
            loss_data  = self.loss_fn(y_true, y_pred)
            loss_phy   = self.compute_physics_loss(reflectance_obs, y_pred)
            total_loss = loss_data + self.lambda_phy * loss_phy

        grads, _ = tf.clip_by_global_norm(
            tape.gradient(total_loss, self.trainable_variables), 1.0
        )
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(loss_data)
        self.phy_loss_tracker.update_state(loss_phy)
        self.rmse_metric.update_state(y_true, y_pred)
        self.mae_metric.update_state(y_true, y_pred)

        return {
            "loss":      self.total_loss_tracker.result(),
            "rmse":      self.rmse_metric.result(),
            "mae":       self.mae_metric.result(),
            "loss_data": self.data_loss_tracker.result(),
            "loss_phy":  self.phy_loss_tracker.result(),
            "lambda":    self.lambda_phy,
            "k_mean":    tf.reduce_mean(self.k),
        }

    def test_step(self, data):
        (x, reflectance_obs), y_true = data
        y_pred     = self.apply_depth_constraint(self.base_model(x, training=False))
        loss_data  = self.loss_fn(y_true, y_pred)
        loss_phy   = self.compute_physics_loss(reflectance_obs, y_pred)
        total_loss = loss_data + self.lambda_phy * loss_phy

        self.total_loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(loss_data)
        self.phy_loss_tracker.update_state(loss_phy)
        self.rmse_metric.update_state(y_true, y_pred)
        self.mae_metric.update_state(y_true, y_pred)

        return {
            "loss":      self.total_loss_tracker.result(),
            "rmse":      self.rmse_metric.result(),
            "mae":       self.mae_metric.result(),
            "loss_data": self.data_loss_tracker.result(),
            "loss_phy":  self.phy_loss_tracker.result(),
        }

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer


# ============================================================================
# 3.  Training callbacks (kept here for completeness; not needed at inference)
# ============================================================================

class LambdaSchedulerCallback(tf.keras.callbacks.Callback):
    """
    Cosine / linear decay of the physics weight (lambda) during training.
    Starts high to enforce physics, then decays so data refines the fit.
    """

    def __init__(
        self,
        pinn_model: PINNWrapper,
        total_epochs: int,
        decay_type: str = "cosine",
    ):
        super().__init__()
        self.pinn_model   = pinn_model
        self.total_epochs = total_epochs
        self.decay_type   = decay_type

    def on_epoch_begin(self, epoch: int, logs=None):
        if not self.pinn_model.adaptive_lambda:
            return
        p = epoch / self.total_epochs
        if self.decay_type == "cosine":
            new_lam = self.pinn_model.lambda_phy_initial * (
                0.1 + 0.9 * (1 + np.cos(p * np.pi)) / 2
            )
        elif self.decay_type == "linear":
            new_lam = self.pinn_model.lambda_phy_initial * (1.0 - 0.9 * p)
        else:
            new_lam = self.pinn_model.lambda_phy_initial
        self.pinn_model.lambda_phy.assign(float(new_lam))


class PhysicsMonitorCallback(tf.keras.callbacks.Callback):
    """Print learned physics parameters every N epochs."""

    def __init__(self, monitor_freq: int = 50):
        super().__init__()
        self.monitor_freq = monitor_freq

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.monitor_freq == 0:
            try:
                k    = self.model.k.numpy()
                R0   = self.model.R0.numpy()
                Rinf = self.model.Rinf.numpy()
                print(f"\n{'='*40}")
                print(f"[Physics Parameters @ Epoch {epoch}]")
                print(f"  k   (attenuation): {np.round(k,    4)}")
                print(f"  R0  (shallow):     {np.round(R0,   4)}")
                print(f"  Rinf(deep):        {np.round(Rinf, 4)}")
                print(f"{'='*40}")
            except AttributeError:
                print("Warning: model has no physics parameters.")


# ============================================================================
# 4.  High-level inference wrapper
# ============================================================================

#: Band order expected by the model
EXPECTED_BANDS = ["B1", "B2", "B3", "B4", "B8", "B8A", "B11", "B12"]

#: Depth clipping defaults
DEPTH_MIN_DEFAULT = 0.0
DEPTH_MAX_DEFAULT = 30.0


class BathymetryInference:
    """
    High-level wrapper that loads the PINN and exposes clean predict methods.

    Parameters
    ----------
    model      : Loaded PINNWrapper instance.
    scaler     : Fitted sklearn scaler, or None.
    depth_min  : Minimum clipping depth (m).
    depth_max  : Maximum clipping depth (m).
    batch_size : Number of pixels per TF inference call.

    Examples
    --------
    >>> infer = load_bathymetry_model('/path/to/model_dir')
    >>> depth_da = infer.predict_xarray(preprocessed_scene_da)  # xr.DataArray (y, x)
    """

    def __init__(
        self,
        model:      PINNWrapper,
        scaler=None,
        depth_min:  float = DEPTH_MIN_DEFAULT,
        depth_max:  float = DEPTH_MAX_DEFAULT,
        batch_size: int   = 2048,
    ):
        self.model      = model
        self.scaler     = scaler
        self.depth_min  = depth_min
        self.depth_max  = depth_max
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Low-level: NumPy in → NumPy out
    # ------------------------------------------------------------------

    def predict_array(self, reflectance_array: np.ndarray) -> np.ndarray:
        """
        Predict depth from a 2-D reflectance array.

        Args:
            reflectance_array : np.ndarray  shape (n_pixels, 8)
                                Band order must match EXPECTED_BANDS.
                                Values should be in [0, 1] (surface reflectance).

        Returns:
            depths : np.ndarray  shape (n_pixels,)  depth in metres.
        """
        X = np.asarray(reflectance_array, dtype=np.float32)
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = self.scaler.transform(X).astype(np.float32)

        # Batch inference to keep GPU/CPU memory under control
        n      = len(X)
        depths = np.empty(n, dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end           = min(start + self.batch_size, n)
            batch_depths  = self.model(X[start:end], training=False).numpy().flatten()
            depths[start:end] = np.clip(batch_depths, self.depth_min, self.depth_max)

        return depths

    # ------------------------------------------------------------------
    # High-level: Xarray scene (band, y, x) → depth (y, x)
    # ------------------------------------------------------------------

    def predict_xarray(self, scene: xr.DataArray) -> xr.DataArray:
        """
        Predict bathymetry for a single S2 scene.

        Args:
            scene : xr.DataArray  dims (band, y, x)
                    Coordinate 'band' must contain the names in EXPECTED_BANDS.
                    Values should be preprocessed reflectances in [0, 1].

        Returns:
            depth : xr.DataArray  dims (y, x)  values in metres.
                    NaN where the input had NaN or zero in all bands.
        """
        # ── Validate band coordinate ─────────────────────────────────────
        missing = [b for b in EXPECTED_BANDS if b not in scene.band.values]
        if missing:
            raise ValueError(
                f"Scene is missing bands: {missing}. "
                f"Expected: {EXPECTED_BANDS}"
            )

        # Reorder to match expected band sequence
        scene = scene.sel(band=EXPECTED_BANDS)

        arr   = scene.values               # (n_bands, y, x)
        h, w  = arr.shape[1], arr.shape[2]

        # ── Flatten to (n_pixels, n_bands) ───────────────────────────────
        pixels = arr.reshape(len(EXPECTED_BANDS), -1).T  # (n_pixels, n_bands)

        # ── Valid pixel mask: no NaN and at least one band non-zero ──────
        valid = ~(np.any(np.isnan(pixels), axis=1) | np.all(pixels == 0, axis=1))

        depth_flat = np.full(h * w, np.nan, dtype=np.float32)

        if valid.any():
            depth_flat[valid] = self.predict_array(pixels[valid])

        depth_2d = depth_flat.reshape(h, w)

        # ── Trim depth_2d to match coordinate lengths (xee off-by-one) ──
        y_len = len(scene.y)
        x_len = len(scene.x)
        depth_2d = depth_2d[:y_len, :x_len]

        return xr.DataArray(
            depth_2d,
            dims=["y", "x"],
            coords={"y": scene.y, "x": scene.x},
            name="bathymetry",
            attrs={
                "units":     "metres",
                "depth_min": self.depth_min,
                "depth_max": self.depth_max,
                "bands":     EXPECTED_BANDS,
            },
        )


# ============================================================================
# 5.  Factory function — the single public entry point
# ============================================================================

def load_bathymetry_model(
    model_dir:  str,
    n_bands:    int   = 8,
    depth_min:  float = DEPTH_MIN_DEFAULT,
    depth_max:  float = DEPTH_MAX_DEFAULT,
    batch_size: int   = 2048,
    verbose:    bool  = True,
) -> BathymetryInference:
    """
    Load the trained PINN bathymetry model from disk.

    Expected files in model_dir
    ---------------------------
    physics_params.json      — dict with keys 'k', 'R0', 'Rinf' (lists of floats)
    pinn_weights.weights.h5  — Keras saved weights
    scaler.pkl               — (optional) sklearn scaler

    Args:
        model_dir  : Path to directory containing the three files above.
        n_bands    : Number of input spectral bands (default 8).
        depth_min  : Minimum depth clipping value (metres).
        depth_max  : Maximum depth clipping value (metres).
        batch_size : Pixels per inference batch.
        verbose    : Print loading progress.

    Returns:
        BathymetryInference  — ready to call .predict_xarray() or .predict_array().

    Raises:
        FileNotFoundError : if physics_params.json or pinn_weights.weights.h5 are missing.
    """
    # ── 1. Physics parameters ────────────────────────────────────────────
    params_path = os.path.join(model_dir, "physics_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"physics_params.json not found in {model_dir}")

    with open(params_path, "r") as f:
        physics_params = json.load(f)

    if verbose:
        print("✓ Physics parameters loaded:")
        print(f"    k    = {np.round(physics_params['k'],    6).tolist()}")
        print(f"    R0   = {np.round(physics_params['R0'],   6).tolist()}")
        print(f"    Rinf = {np.round(physics_params['Rinf'], 6).tolist()}")

    # ── 2. Build PINN skeleton ───────────────────────────────────────────
    if verbose:
        print("\nBuilding PINN model skeleton...")

    base_model = create_transformer_model(
        input_shape=(n_bands,), n_blocks=3, embed_dim=64, n_heads=4
    )
    pinn = PINNWrapper(
        base_model=base_model,
        n_bands=n_bands,
        lambda_phy=0.1,
        depth_min=depth_min,
        depth_max=depth_max,
        adaptive_lambda=False,   # inference only — no scheduler
        init_k=np.asarray(physics_params["k"],    dtype=np.float32),
        init_R0=np.asarray(physics_params["R0"],  dtype=np.float32),
        init_Rinf=np.asarray(physics_params["Rinf"], dtype=np.float32),
    )

    # Build graph with a dummy forward pass
    _ = pinn(tf.zeros((1, n_bands), dtype=tf.float32), training=False)

    # ── 3. Restore weights ───────────────────────────────────────────────
    weights_path = os.path.join(model_dir, "pinn_weights.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"pinn_weights.weights.h5 not found in {model_dir}")

    pinn.load_weights(weights_path)
    if verbose:
        print(f"✓ Weights restored from: {weights_path}")

    # ── 4. Physics parameter check ───────────────────────────────────────
    if verbose:
        for name, saved, loaded in [
            ("k",    physics_params["k"],    pinn.k.numpy()),
            ("R0",   physics_params["R0"],   pinn.R0.numpy()),
            ("Rinf", physics_params["Rinf"], pinn.Rinf.numpy()),
        ]:
            diff = np.max(np.abs(np.asarray(loaded) - np.asarray(saved)))
            print(f"  {name:4s} max_diff = {diff:.2e}")

    # ── 5. Scaler (optional) ─────────────────────────────────────────────
    scaler      = None
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if verbose:
            print("✓ Scaler loaded")
    else:
        if verbose:
            print("⚠  No scaler found — inputs must already be normalised.")

    # ── 6. Wrap and return ───────────────────────────────────────────────
    inference = BathymetryInference(
        model=pinn,
        scaler=scaler,
        depth_min=depth_min,
        depth_max=depth_max,
        batch_size=batch_size,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL READY FOR INFERENCE")
        print("=" * 60)
        print(f"  Usage : infer.predict_xarray(scene_da)")
        print(f"       or infer.predict_array(array)  — (n_pixels, {n_bands})")
        print(f"  Bands : {EXPECTED_BANDS}")
        print(f"  Depth : {depth_min} – {depth_max} m")
        print("=" * 60)

    return inference
