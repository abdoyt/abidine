# ML Package for Image Denoising

This package provides a complete pipeline for training a denoising autoencoder on coronary medical images.

## Structure

```
ml/
├── data/
│   ├── dataset.py      # Dataset classes for loading images
│   └── preprocess.py   # Preprocessing and noise simulation
└── models/
    └── autoencoder.py  # CNN autoencoder architecture
```

## Features

### Data Loading (`ml/data/dataset.py`)
- **Multi-format support**: PNG, JPEG, and DICOM
- **DenoiseDataset**: PyTorch Dataset that loads images and creates noisy/clean pairs on-the-fly
- **Automatic preprocessing**: Resizing, normalization, and noise simulation
- **Helper functions**: `create_dataloaders()` for easy setup

### Preprocessing (`ml/data/preprocess.py`)
- **Normalization**: Scale images to [0, 1] range
- **Resizing**: Support for 256×256 or 512×512 target sizes
- **Noise simulation**:
  - Poisson noise (signal-dependent, simulates photon counting)
  - Gaussian noise (additive)
  - Combined low-dose simulation
- **On-the-fly generation**: Create training pairs dynamically

### Model Architecture (`ml/models/autoencoder.py`)
- **U-Net style autoencoder**: Encoder-decoder with skip connections
- **Lightweight design**: ~1.2M parameters (base_channels=32)
- **Architecture details**:
  - 4 downsampling stages (max pooling)
  - Bottleneck with 2 conv blocks
  - 4 upsampling stages (transposed conv)
  - Skip connections for detail preservation
  - Sigmoid output for [0, 1] range

## Usage

### Training

```bash
# Using default configuration
python train_denoise.py

# Using custom configuration
python train_denoise.py --config configs/my_config.yaml

# Override specific parameters
python train_denoise.py --epochs 100 --lr 0.0005 --batch-size 16

# Train on CPU
python train_denoise.py --device cpu
```

### Configuration

See `configs/denoise_example.yaml` for all available options:
- Data parameters (paths, batch size, image size)
- Noise simulation (Poisson scale, Gaussian std)
- Model architecture (channels, layers)
- Training hyperparameters (epochs, learning rate)
- Output paths (checkpoints, logs)

### Data Preparation

Organize your images in a directory structure:

```
data/
├── train/
│   ├── image001.png
│   ├── image002.jpg
│   ├── image003.dcm
│   └── ...
└── val/  (optional)
    ├── image001.png
    └── ...
```

The dataset will automatically:
1. Find all supported image files
2. Load and convert to grayscale
3. Resize to target size
4. Generate noisy versions on-the-fly

### Monitoring Training

**TensorBoard**:
```bash
tensorboard --logdir artifacts/logs
```

**CSV logs**: Check `artifacts/training_log.csv` for detailed metrics

### Output

Training produces:
- `artifacts/denoiser.pt`: Best model checkpoint
- `artifacts/checkpoint_epoch_N.pt`: Periodic checkpoints
- `artifacts/logs/`: TensorBoard event files
- `artifacts/training_log.csv`: Training metrics in CSV format

## Metrics

The training script tracks:
- **Loss**: MSE between predicted and clean images
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better, 0-1 range)

## Requirements

Install dependencies:
```bash
pip install -r requirements-ml.txt
```

Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- pydicom >= 2.4.0 (for DICOM support)
- scikit-image >= 0.21.0 (for SSIM)
- tensorboard >= 2.14.0
- pyyaml >= 6.0

## Example: Quick Start

```python
# 1. Prepare your data in data/train/
# 2. Create or modify configs/denoise_example.yaml
# 3. Train the model
python train_denoise.py --epochs 50 --batch-size 8

# 4. Load trained model for inference
import torch
from ml.models.autoencoder import create_model

model = create_model()
checkpoint = torch.load('artifacts/denoiser.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 5. Denoise an image
with torch.no_grad():
    denoised = model(noisy_image)
```

## Tips

- **Small dataset**: Use data augmentation or train with more noise variations
- **GPU memory**: Reduce batch size or image size if OOM errors occur
- **Overfitting**: Increase weight decay or reduce model size
- **Convergence**: Monitor both PSNR and SSIM, not just loss
- **Learning rate**: Use automatic scheduling (already enabled)
