# ML Package Setup Guide

This guide will help you get started with the denoising training pipeline.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements-ml.txt
```

This installs:
- PyTorch & torchvision (deep learning framework)
- NumPy (numerical computing)
- Pillow (image processing)
- pydicom (DICOM medical image format)
- scikit-image (image metrics like SSIM)
- TensorBoard (training visualization)
- PyYAML (configuration files)
- tqdm (progress bars)

### 2. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
.
├── ml/                           # ML package
│   ├── data/
│   │   ├── dataset.py           # Dataset classes
│   │   └── preprocess.py        # Image preprocessing & noise simulation
│   ├── models/
│   │   └── autoencoder.py       # CNN autoencoder architecture
│   ├── generate_test_data.py    # Generate synthetic test images
│   ├── inference_example.py     # Example inference script
│   └── README.md                # Detailed ML documentation
├── configs/
│   └── denoise_example.yaml     # Training configuration
├── train_denoise.py             # Main training script
├── requirements-ml.txt          # Python dependencies
└── artifacts/                   # Output directory (created during training)
    ├── denoiser.pt             # Best model checkpoint
    ├── checkpoint_epoch_*.pt   # Periodic checkpoints
    ├── logs/                   # TensorBoard logs
    └── training_log.csv        # Training metrics
```

## Quick Start

### Option 1: Using Synthetic Test Data

1. **Generate test images**:
```bash
python ml/generate_test_data.py --output-dir data/train --num-images 50 --val
```

2. **Train the model**:
```bash
python train_denoise.py --config configs/denoise_example.yaml
```

3. **Monitor training** (in another terminal):
```bash
tensorboard --logdir artifacts/logs
```
Then open http://localhost:6006

### Option 2: Using Your Own Images

1. **Organize your images**:
```
data/
├── train/
│   ├── image001.png
│   ├── image002.jpg
│   ├── image003.dcm
│   └── ...
└── val/  (optional)
    └── ...
```

Supported formats: PNG, JPEG, DICOM

2. **Update configuration** (optional):

Edit `configs/denoise_example.yaml` and adjust paths:
```yaml
data:
  train_dir: "path/to/your/train/images"
  val_dir: "path/to/your/val/images"  # optional
```

3. **Train**:
```bash
python train_denoise.py --config configs/denoise_example.yaml
```

## Training Configuration

### Key Parameters

Edit `configs/denoise_example.yaml` or override via command line:

**Data**:
- `image_size`: 256 or 512 (higher = better quality, more memory)
- `batch_size`: 8 (reduce if out of memory)
- `num_workers`: 2 (parallel data loading)

**Noise Simulation**:
- `poisson_scale`: 5.0 (higher = less noise)
- `gaussian_std`: 0.05 (additive noise level)

**Model**:
- `base_channels`: 32 (32 = lightweight ~1.2M params, 64 = larger ~4.8M params)

**Training**:
- `epochs`: 50 (training duration)
- `learning_rate`: 0.001 (start learning rate)
- `device`: "auto" (auto-detect GPU/CPU)

### Command-Line Overrides

```bash
# Train for more epochs
python train_denoise.py --epochs 100

# Use larger batch size
python train_denoise.py --batch-size 16

# Custom learning rate
python train_denoise.py --lr 0.0005

# Force CPU training
python train_denoise.py --device cpu

# 512x512 images
python train_denoise.py --image-size 512

# Combine multiple overrides
python train_denoise.py --epochs 100 --batch-size 16 --image-size 512
```

## Training Output

During training, you'll see:
- Progress bars with loss, PSNR, SSIM
- Periodic checkpoint saves
- Best model saved when validation improves

### Metrics

- **Loss (MSE)**: Mean squared error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better, typical: 25-35 dB)
- **SSIM**: Structural Similarity Index 0-1 (higher is better, typical: 0.7-0.95)

### Files Created

- `artifacts/denoiser.pt`: Best model (lowest validation loss)
- `artifacts/checkpoint_epoch_N.pt`: Periodic checkpoints
- `artifacts/logs/`: TensorBoard event files
- `artifacts/training_log.csv`: All metrics in CSV format

## Using Trained Models

### Inference Example

```bash
python ml/inference_example.py \
  --checkpoint artifacts/denoiser.pt \
  --input path/to/noisy_image.png \
  --output path/to/denoised_image.png
```

### In Python

```python
import torch
from ml.models.autoencoder import create_model
from ml.data.preprocess import normalize_image, resize_image
from PIL import Image
import numpy as np

# Load model
model = create_model()
checkpoint = torch.load('artifacts/denoiser.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
img = Image.open('input.png').convert('L')
img_array = np.array(img, dtype=np.float32)
img_normalized = normalize_image(img_array)
img_resized = resize_image(img_normalized, 256)

# Convert to tensor
img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# Denoise
with torch.no_grad():
    denoised_tensor = model(img_tensor)

# Convert back to image
denoised = denoised_tensor.squeeze().numpy()
denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8), mode='L')
denoised_pil.save('output.png')
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: "CUDA out of memory" or system freezing

**Solutions**:
1. Reduce batch size: `--batch-size 4` or `--batch-size 2`
2. Use smaller images: `--image-size 256`
3. Use smaller model: Edit config, set `base_channels: 16`
4. Train on CPU: `--device cpu` (slower but no memory limit)

### Poor Results

**Symptoms**: Low PSNR, blurry output

**Solutions**:
1. More training data (at least 50-100 images recommended)
2. Longer training: `--epochs 100`
3. Larger model: Edit config, set `base_channels: 64`
4. Better data quality (ensure images are clear and aligned)
5. Adjust noise levels to match your target use case

### Slow Training

**Solutions**:
1. Use GPU: Install CUDA-enabled PyTorch
2. Increase batch size (if memory allows): `--batch-size 16`
3. Reduce image size: `--image-size 256`
4. Use more workers: Edit config, set `num_workers: 4`

### ImportError: No module named 'X'

**Solution**: Install dependencies
```bash
pip install -r requirements-ml.txt
```

## Advanced Usage

### Resume Training

Modify `train_denoise.py` to load checkpoint in the train() function:
```python
# After creating model
if os.path.exists('artifacts/checkpoint_epoch_10.pt'):
    checkpoint = torch.load('artifacts/checkpoint_epoch_10.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### Custom Model Architecture

Edit `ml/models/autoencoder.py` and adjust:
- Number of encoder/decoder stages
- Convolution kernel sizes
- Activation functions
- Skip connection strategy

### Custom Noise Model

Edit `ml/data/preprocess.py` to implement different noise types:
- Speckle noise
- Salt & pepper
- Real noise profiles from calibration data

### Data Augmentation

Modify `DenoiseDataset.__getitem__()` in `ml/data/dataset.py`:
- Random flips/rotations
- Random crops
- Intensity variations

## Support

For detailed documentation, see:
- [ml/README.md](ml/README.md) - Comprehensive ML package docs
- [train_denoise.py](train_denoise.py) - Main training script with detailed comments

For issues or questions:
1. Check training logs in `artifacts/training_log.csv`
2. Review TensorBoard plots
3. Verify data loading with a small subset first
