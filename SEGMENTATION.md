# Coronary Artery Segmentation Module (Optional)

This document describes the **optional** segmentation module (Module B) that has been scaffolded and is ready for development.

## Overview

The segmentation module provides a complete pipeline for coronary artery segmentation in angiography images using a lightweight 2D U-Net architecture. It is designed to work with denoised images (output from Module A) and produces colorized overlays for visualization in the Streamlit app.

## Module Status

**OPTIONAL**: This module is scaffolded and ready to run, but development may continue based on project needs. All code validates using synthetic datasets.

## Components

### 1. U-Net Model (`ml/models/unet.py`)

- **Class**: `UNet2D`
- **Architecture**: Lightweight 2D U-Net with 4 downsampling blocks and 4 upsampling blocks
- **Input**: Single-channel grayscale images (1, H, W)
- **Output**: Logits for binary segmentation (2 classes: background, artery)
- **Features**:
  - Batch normalization
  - Skip connections
  - Configurable bilinear upsampling
  - Factory function `create_unet()` for easy instantiation

### 2. Mask Dataset (`ml/data/mask_dataset.py`)

**Class**: `MaskDataset`

Handles image-mask pair loading and processing:

- **File formats**: Supports .npy and .png
- **Data augmentation**:
  - 90° rotations
  - Horizontal/vertical flips
  - Brightness adjustment (0.8-1.2x)
  - Contrast adjustment (0.8-1.2x)
- **Class balancing**:
  - Weighted sampling based on positive pixel count
  - `WeightedRandomSampler` integration
- **Normalization**: Automatic float32 normalization to [0, 1]

**Synthetic Dataset Generation**: `create_synthetic_mask_dataset()`
- Generates circular/elliptical vessel-like patterns
- Creates paired images and masks
- Configurable number of samples and image size
- Deterministic via random seed

### 3. Training Script (`ml/train_segment.py`)

Complete training pipeline with:

**Features**:
- Argparse configuration for all hyperparameters
- Combined Dice + BCE loss (50/50 weighted by default)
- Model checkpointing (best model + periodic)
- Learning rate scheduling (ReduceLROnPlateau)
- Synthetic dataset support built-in
- Config saving for reproducibility

**Usage**:
```bash
# Train on synthetic data (default)
python ml/train_segment.py --epochs 20 --batch-size 8

# Train on custom dataset
python ml/train_segment.py \
  --image-dir ./data/images \
  --mask-dir ./data/masks \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 0.001

# Key arguments
--synthetic          # Use synthetic dataset (default: True)
--num-synthetic      # Number of synthetic samples (default: 50)
--output-dir         # Checkpoint directory (default: ./checkpoints/segment)
--batch-size         # Batch size (default: 8)
--learning-rate      # Learning rate (default: 1e-3)
--epochs             # Number of epochs (default: 20)
--augment            # Enable augmentation (default: True)
--balance            # Enable class balancing (default: True)
--device             # cuda/cpu (auto-detected)
--seed               # Random seed (default: 42)
```

**Outputs**:
- `best_model.pth`: Best performing model
- `checkpoint_epoch_XXX.pth`: Periodic checkpoints
- `final_model.pth`: Final trained model
- `config.json`: Configuration snapshot

### 4. Inference Script (`ml/inference_segment.py`)

Produces segmentation masks and colorized overlays:

**Features**:
- Loads trained U-Net model from checkpoint
- Processes images (batch or single)
- Generates artery probability masks (0-1)
- Creates colorized overlays (red arteries on original)
- Saves outputs in multiple formats (.npy for processing, .png for visualization)
- Confidence threshold control

**Usage**:
```bash
# Run inference on denoised images
python ml/inference_segment.py \
  --model ./checkpoints/segment/best_model.pth \
  --input-dir ./data/denoised \
  --output-dir ./data/segmented \
  --threshold 0.5 \
  --alpha 0.5

# Key arguments
--model              # Path to trained model checkpoint (required)
--input-dir          # Input images directory (default: ./data/denoised)
--output-dir         # Output directory (default: ./data/segmented)
--image-pattern      # File pattern (default: *.npy)
--threshold          # Artery confidence threshold (default: 0.5)
--alpha              # Overlay opacity (default: 0.5)
--device             # cuda/cpu (auto-detected)
```

**Outputs per image**:
- `{image}_mask.npy`: Artery probability mask (H, W) values in [0, 1]
- `{image}_mask.png`: Probability mask as grayscale PNG
- `{image}_overlay.png`: Colorized overlay (red arteries on original)
- `inference_summary.json`: Processing summary

## Data Format

### Expected Directory Structure

```
data/
├── denoised/           # Input (denoised images from Module A)
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
├── masks/              # Ground truth (optional, for training)
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
└── segmented/          # Output (inference results)
    ├── image_001_mask.npy
    ├── image_001_mask.png
    ├── image_001_overlay.png
    └── ...
```

### File Formats

- **Images**: .npy (numpy arrays) or .png (8-bit grayscale)
  - Shape: (H, W) for 2D single-channel
  - Value range: [0, 255] (uint8) or [0, 1] (float32)

- **Masks**: .npy or .png
  - Binary: 0 (background) or 1 (artery)
  - Multi-class: 0, 1, 2, ... (supported but not used yet)

## Streamlit Integration

The inference script generates colorized overlays optimized for Streamlit display:

```python
from PIL import Image
import streamlit as st

# Load and display overlay
overlay = Image.open("path/to/image_overlay.png")
st.image(overlay, caption="Artery Segmentation Overlay")

# Load probability mask
import numpy as np
mask = np.load("path/to/image_mask.npy")
st.image(mask, caption="Artery Probability Mask", clamp=True)
```

## Development Notes

- **Optional**: Can be skipped entirely in production
- **Validation**: All code validates on synthetic datasets
- **Dependencies**: PyTorch, PIL, OpenCV, numpy
- **GPU**: Supports CUDA if available, falls back to CPU
- **Scalability**: Lightweight U-Net suitable for real-time inference on typical hardware

## Next Steps

When ready for production use:
1. Train on real coronary artery datasets (e.g., DRIVE, STARE, or proprietary datasets)
2. Fine-tune hyperparameters for specific imaging modalities
3. Implement advanced augmentation (elastic deformation, GAN-based)
4. Add ensemble methods or post-processing (CRF, morphological operations)
5. Optimize for deployment (quantization, model compression)
6. Integrate with Streamlit app for real-time prediction

## Testing

Run the training script with synthetic data to validate the pipeline:

```bash
python ml/train_segment.py --synthetic --num-synthetic 20 --epochs 2
```

This will:
1. Generate 20 synthetic image-mask pairs
2. Train for 2 epochs
3. Save checkpoints to `./checkpoints/segment/`

Then run inference:

```bash
# Copy synthetic images for inference
cp checkpoints/segment/synthetic_data/images/* data/denoised/

python ml/inference_segment.py \
  --model ./checkpoints/segment/best_model.pth \
  --input-dir ./data/denoised \
  --output-dir ./data/results
```

## References

- Original U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Dice Loss: Milletari et al., "The Dice Similarity Coefficient for Image Segmentation" (2016)
- Coronary Segmentation: Common benchmark datasets (DRIVE, STARE)
