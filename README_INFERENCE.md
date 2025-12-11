# Denoising Autoencoder Inference CLI

A comprehensive command-line interface for running inference on denoising autoencoder models. Supports PNG, JPEG, and DICOM image formats with optional PSNR/SSIM metrics calculation.

## Features

- **Multiple Image Formats**: Supports PNG, JPEG, DICOM, and other standard image formats
- **Single & Batch Processing**: Process individual images or entire folders
- **Model Checkpoint Loading**: Load saved autoencoder weights for inference
- **Metrics Calculation**: Compute PSNR, SSIM, MSE, and MAE metrics
- **Comparison Generation**: Generate side-by-side before/after comparisons
- **Flexible Configuration**: Customizable image sizes, batch processing, and device selection
- **Comprehensive Logging**: Clear console output for headless operation

## Installation

### Dependencies

Install the required Python dependencies:

```bash
pip install -r requirements-ml.txt
```

Core dependencies include:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- PIL (Pillow) >= 8.3.0
- numpy >= 1.21.0
- scikit-image >= 0.18.0
- pydicom >= 2.2.0

Optional dependencies:
- matplotlib >= 3.4.0 (for comparison generation)
- opencv-python >= 4.5.0 (for advanced image processing)
- tqdm >= 4.62.0 (for progress bars)

## Quick Start

### Basic Usage

```bash
# Process a single image
python inference_denoise.py --input image.png --output denoised.png --checkpoint model.pth

# Process an entire folder
python inference_denoise.py --input-folder images/ --output-folder denoised/ --checkpoint model.pth

# Process with metrics and comparison
python inference_denoise.py --input image.png --output denoised.png --metrics --comparison --checkpoint model.pth
```

## Command Line Arguments

### Input/Output
- `--input`: Path to input image file (single image mode)
- `--input-folder`: Path to folder containing images (batch mode)
- `--output`: Path to output denoised image (required for single image mode)
- `--output-folder`: Path to output folder (required for batch mode)
- `--comparison`: Generate side-by-side comparison images
- `--metrics`: Compute PSNR/SSIM metrics

### Model Configuration
- `--checkpoint`: Path to model checkpoint (required)
- `--config`: Path to model configuration file (JSON)
- `--image-size`: Target image size as "width height" (e.g., "512 512")

### Processing Options
- `--recursive`: Process folders recursively (for batch mode)
- `--batch-size`: Batch size for processing (default: 1)
- `--device`: Compute device - "auto", "cpu", or "cuda" (default: "auto")
- `--output-dir`: Base output directory for results (default: "inference_results")

### Logging
- `--log-level`: Logging level - "DEBUG", "INFO", "WARNING", "ERROR" (default: "INFO")
- `--quiet`: Suppress non-error output

## Examples

### Single Image Processing

```bash
# Basic denoising
python inference_denoise.py \
    --input test_image.png \
    --output denoised_test_image.png \
    --checkpoint autoencoder_epoch50.pth

# With metrics and custom size
python inference_denoise.py \
    --input medical_scan.dcm \
    --output denoised_scan.png \
    --checkpoint model.pth \
    --image-size 512 512 \
    --metrics
```

### Batch Processing

```bash
# Process entire folder recursively
python inference_denoise.py \
    --input-folder /path/to/images/ \
    --output-folder /path/to/denoised/ \
    --checkpoint model.pth \
    --recursive \
    --batch-size 4

# Process with metrics and comparisons
python inference_denoise.py \
    --input-folder images/ \
    --output-folder results/ \
    --checkpoint model.pth \
    --metrics \
    --comparison
```

### Advanced Usage

```bash
# Force CPU processing with custom output directory
python inference_denoise.py \
    --input image.png \
    --output result.png \
    --checkpoint model.pth \
    --device cpu \
    --output-dir custom_results/ \
    --log-level DEBUG

# Using model configuration
python inference_denoise.py \
    --input image.png \
    --output denoised.png \
    --checkpoint model.pth \
    --config model_config.json \
    --image-size 256 256
```

## Output Files

The CLI generates various output files depending on the options used:

### Single Image Mode
- **Denoised Image**: The main output (specified by `--output`)
- **Comparison Image**: If `--comparison` is used (side-by-side view)
- **Metrics**: Printed to console and optionally saved

### Batch Mode
- **Output Folder**: All denoised images
- **Comparison Folder**: If `--comparison` is used
- **Summary JSON**: `batch_summary.json` with processing results and metrics

## Model Checkpoint Format

The CLI supports various checkpoint formats:

1. **Full Training Checkpoint**: Contains model_state_dict, optimizer_state_dict, epoch, loss, and config
2. **State Dict Only**: Just the model parameters
3. **Legacy Format**: Various checkpoint formats with different key names

Example checkpoint loading:
```python
from ml.utils.model_utils import ModelLoader

loader = ModelLoader()
model = loader.load_model_with_checkpoint("path/to/checkpoint.pth", model_config)
```

## Metrics Calculation

### Available Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in decibels
- **SSIM (Structural Similarity Index)**: Measures structural similarity (0-1 scale)
- **MSE (Mean Squared Error)**: Average squared differences
- **MAE (Mean Absolute Error)**: Average absolute differences

### Batch Metrics
For batch processing, the CLI calculates:
- Individual metrics for each image
- Aggregate statistics (mean, std, min, max)
- Overall processing time and speed

## Testing

Run the test suite to verify functionality:

```bash
python test_inference.py
```

This tests:
- Single image processing
- Batch folder processing
- Metrics calculation
- Different image format support

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use `--device cpu` to run on CPU
   - Reduce `--batch-size`
   - Use smaller `--image-size`

2. **Unsupported Image Format**
   - Supported formats: PNG, JPEG, DICOM, BMP, TIFF
   - Ensure file has correct extension

3. **Model Loading Errors**
   - Check checkpoint file exists and is readable
   - Verify model configuration matches checkpoint
   - Try loading with `--log-level DEBUG`

4. **Dependencies Missing**
   - Install all requirements: `pip install -r requirements-ml.txt`
   - For DICOM support: `pip install pydicom`
   - For metrics: `pip install scikit-image`

### Debug Mode

Enable detailed logging:
```bash
python inference_denoise.py --log-level DEBUG --input image.png --output result.png --checkpoint model.pth
```

## Performance Tips

1. **GPU Usage**: Use CUDA when available for faster processing
2. **Batch Processing**: Increase batch size for multiple images
3. **Image Size**: Resize large images to reduce processing time
4. **Memory Management**: Process in smaller batches for large datasets

## Integration

The CLI can be integrated into larger workflows:

```python
from inference_denoise import InferenceCLI
from ml.data.preprocessing import ImageStandardizer

# Programmatic usage
cli = InferenceCLI()
cli.load_model("model.pth")
cli.standardizer = ImageStandardizer(target_size=(256, 256))

result = cli.process_single_image("input.png", "output.png", compute_metrics=True)
```

## License

This inference CLI is part of the denoising autoencoder project.