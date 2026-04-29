# MelHilbertLDS

Audio classification framework with Mel spectrogram, Hilbert curve spatial mapping, and Low Discrepancy Sequence (LDS) sampling.

## Core Modules

### Training

- **train.py** — Main training script.
  - Supports multiple feature types: `Mel`, `MelLDS`, `MelHilbert`, `MelHilbertLDS`, `MelHilbertTime`, `MelHilbertTimeLDS`, `SignalHilbert`, `SignalHilbertLDS`, `Signal`, `SignalLDS`.
  - Supports multiple datasets: `ESC50-human`, `CoughDataset`, `cough-speech-sneeze`.
  - Supports multiple model backends: `naive`, `mobile`, `pann`.
  - Logs to TensorBoard and saves the best checkpoint.
  - Example:
    ```bash
    python train.py --type MelHilbertLDS --dataset ESC50-human --model_backend mobile --seed 42 --hop_length 64
    ```

- **train_multi_dataset.sh** — Batch training script.
  - Iterates over all combinations of backends, datasets, feature types, and seeds.
  - Usage:
    ```bash
    bash train_multi_dataset.sh [DATE_PREFIX]
    ```

### Data Loading & Preprocessing

- **dataloaders.py** — Dataset loaders and feature generators.
  - `CSSDataset`: loads audio or precomputed Mel features, applies data augmentation, and generates 2D feature maps.
  - Supports LDS frame extraction, Hilbert curve mapping, and interpolation.
  - `get_data_loader()`: returns a PyTorch DataLoader with optional weighted sampling.

- **preprocess_mel.py** — Precompute Mel spectrograms.
  - Batch-converts audio files to `.npy` Mel spectrograms for faster training.
  - Usage:
    ```bash
    python preprocess_mel.py --dataset ESC50-human --hop_length 64
    ```

### Models

- **models_naive.py** — Lightweight naive CNN backend.
- **models_mobilenet.py** — MobileNetV3-based backend with inverted residuals and squeeze-and-excitation blocks.
- **models_pann.py** — PANNs-style CNN backend for audio pattern recognition.

Each backend provides a `get_model(num_classes, type, **kwargs)` factory function.

### Analysis & Visualization

- **analyze_results.py** — Experiment result analyzer.
  - Reads TensorBoard logs, computes mean/std of metrics across seeds.
  - Generates confusion matrices, classification reports, and Markdown summary tables.
  - Usage:
    ```bash
    python analyze_results.py --experiment_dir results/260414_mobile_esc50
    ```

- **contribution_analysis.py** — Factorial contribution analysis.
  - Evaluates the individual and combined contributions of Mel, Hilbert, LDS, and Signal components.
  - Generates summary tables and statistics.

- **illustrate_1.py** — Signal reconstruction comparison (Figure 6).
  - Compares LDS, Uniform, and Random sampling on synthetic signals (types A/B/C).

- **illustrate_2.py** — Mel spectrogram reconstruction comparison (Figure 7).
  - Compares sampling methods on Mel spectrograms using MSE, PSNR, SSIM, and Frobenius distance.

- **illustrate_3.py** — Local continuity visualization (Figure 8).
  - Compares Hilbert curve mapping vs. raster scan mapping by computing max consecutive sequence lengths within a convolution kernel window.

## Quick Start

1. **Precompute Mel spectrograms** (recommended):
   ```bash
   python preprocess_mel.py --dataset ESC50-human --hop_length 64
   ```

2. **Train a single model**:
   ```bash
   python train.py --type MelHilbertLDS --dataset ESC50-human --model_backend mobile --seed 42
   ```

3. **Run batch experiments**:
   ```bash
   bash train_multi_dataset.sh 260414
   ```

4. **Analyze results**:
   ```bash
   python analyze_results.py --experiment_dir results/260414_mobile_esc50
   ```
