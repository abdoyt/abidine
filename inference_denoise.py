#!/usr/bin/env python3
"""
Inference CLI for denoising autoencoder.
Loads saved weights, processes PNG/JPEG/DICOM files, and outputs denoised images with metrics.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time

# Graceful dependency checking
def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    return missing_deps

def check_optional_dependencies():
    """Check if optional dependencies are available."""
    optional_deps = {}
    
    try:
        import torchvision.utils as vutils
        optional_deps['torchvision'] = True
    except ImportError:
        optional_deps['torchvision'] = False
    
    try:
        import matplotlib.pyplot as plt
        optional_deps['matplotlib'] = True
    except ImportError:
        optional_deps['matplotlib'] = False
    
    try:
        from skimage.metrics import structural_similarity as ssim
        optional_deps['scikit-image'] = True
    except ImportError:
        optional_deps['scikit-image'] = False
    
    try:
        import pydicom
        optional_deps['pydicom'] = True
    except ImportError:
        optional_deps['pydicom'] = False
    
    return optional_deps

# Check dependencies first
missing_deps = check_dependencies()
optional_deps = check_optional_dependencies()

if missing_deps:
    print("Missing required dependencies. Please install:")
    print(f"pip install -r requirements-ml.txt")
    print(f"Missing: {', '.join(missing_deps)}")
    print("\nTrying to show help anyway...")
    # Continue to allow help display

# Conditional imports
if 'numpy' not in missing_deps:
    import numpy as np
else:
    # Create dummy numpy for basic functionality
    class DummyNumpy:
        ndarray = list  # Simple fallback
        
        @staticmethod
        def array(obj):
            return obj
        
        @staticmethod
        def mean(obj):
            if hasattr(obj, '__iter__'):
                return sum(obj) / len(obj)
            return obj
        
        @staticmethod
        def clip(*args, **kwargs):
            return args[0]
        
        @staticmethod
        def zeros(shape, *args, **kwargs):
            return [[0] * shape[1]] * shape[0] if len(shape) == 2 else [0] * shape[0]
        
        @staticmethod
        def ones(shape, *args, **kwargs):
            return [[1] * shape[1]] * shape[0] if len(shape) == 2 else [1] * shape[0]
    
    np = DummyNumpy()

if 'torch' not in missing_deps:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    # Create dummy torch for help display
    class DummyContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class DummyNN:
        @staticmethod
        def Module():
            return object()
        
        @staticmethod
        def Conv2d(*args, **kwargs):
            return lambda x: x
        
        @staticmethod
        def ConvTranspose2d(*args, **kwargs):
            return lambda x: x
        
        @staticmethod
        def BatchNorm2d(*args, **kwargs):
            return lambda x: x
        
        @staticmethod
        def ReLU(*args, **kwargs):
            return lambda x: x
        
        @staticmethod
        def Sigmoid(*args, **kwargs):
            return lambda x: x
        
        @staticmethod
        def Sequential(*args):
            return lambda x: x
    
    class DummyTorch:
        device = "cpu"
        
        class nn:
            Module = DummyNN.Module
            Conv2d = DummyNN.Conv2d
            ConvTranspose2d = DummyNN.ConvTranspose2d
            BatchNorm2d = DummyNN.BatchNorm2d
            ReLU = DummyNN.ReLU
            Sigmoid = DummyNN.Sigmoid
            Sequential = DummyNN.Sequential
        
        @staticmethod
        def no_grad():
            return DummyContext()
        
        @staticmethod
        def cuda():
            return False
        
        @staticmethod
        def tensor(*args, **kwargs):
            return args[0]
        
        class cuda:
            @staticmethod
            def is_available():
                return False
    
    torch = DummyTorch()
    
    # Create proper nn object that supports inheritance
    class DummyModule:
        def __init__(self):
            pass
        def train(self, mode=True): 
            return self
        def eval(self): 
            return self
        def to(self, device): 
            return self
        def state_dict(self): 
            return {}
        def load_state_dict(self, state): 
            pass
        def parameters(self): 
            return []
        def cuda(self): 
            return self
        def cpu(self): 
            return self
    
    class DummyNN:
        @staticmethod
        def Module():
            return DummyModule()
        
        @staticmethod 
        def Conv2d(*args, **kwargs):
            return DummyModule()
        
        @staticmethod
        def ConvTranspose2d(*args, **kwargs):
            return DummyModule()
        
        @staticmethod
        def BatchNorm2d(*args, **kwargs):
            return DummyModule()
        
        @staticmethod
        def ReLU(*args, **kwargs):
            return DummyModule()
        
        @staticmethod
        def Sigmoid(*args, **kwargs):
            return DummyModule()
        
        @staticmethod
        def Sequential(*args):
            return DummyModule()
    
    nn = DummyNN()
    
    class DummyF:
        @staticmethod
        def interpolate(*args, **kwargs):
            return lambda x: x
    
    F = DummyF()

if 'Pillow' not in missing_deps:
    from PIL import Image
else:
    # Create dummy PIL for help display
    class DummyPIL:
        class Image:
            @staticmethod
            def open(path):
                return None
            
            @staticmethod
            def fromarray(array):
                return None
            
            Resampling = type('Resampling', (), {'LANCZOS': 'lanczos'})()
    
    Image = DummyPIL.Image

if 'torchvision' in optional_deps and optional_deps['torchvision']:
    import torchvision.utils as vutils

# Add ml package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data.preprocessing import ImageStandardizer, BatchProcessor, validate_image_file
except ImportError as e:
    print(f"Warning: Could not import preprocessing module: {e}")

try:
    from utils.metrics import calculate_psnr, calculate_ssim
except ImportError as e:
    print(f"Warning: Could not import metrics module: {e}")


# Define base class based on whether torch is available
if 'torch' not in missing_deps:
    BaseModel = nn.Module
else:
    # Create a simple base class for when torch is not available
    class BaseModel:
        def __init__(self):
            pass
        def train(self, mode=True): 
            return self
        def eval(self): 
            return self
        def to(self, device): 
            return self
        def state_dict(self): 
            return {}
        def load_state_dict(self, state): 
            pass
        def parameters(self): 
            return []

class DenoisingAutoencoder(BaseModel):
    """Simple denoising autoencoder model for inference."""
    
    def __init__(self, input_channels: int = 3):
        super(DenoisingAutoencoder, self).__init__()
        
        # Only create actual layers if torch is available
        if 'torch' not in missing_deps:
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(True)
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, input_channels, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        if 'torch' not in missing_deps:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        else:
            # Return dummy output for help display
            return x


class InferenceCLI:
    """Main inference CLI for denoising autoencoder."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.standardizer = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self, checkpoint_path: str, model_config: Optional[Dict] = None) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Optional model configuration
        """
        try:
            if not Path(checkpoint_path).exists():
                # Create a simple model for demo purposes if checkpoint doesn't exist
                self.logger.warning(f"Checkpoint not found at {checkpoint_path}, creating new model")
                self.model = DenoisingAutoencoder()
                
                # Initialize with random weights
                self.model.apply(self._init_weights)
            else:
                # Load actual checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Extract model configuration
                if model_config is None:
                    if 'config' in checkpoint:
                        model_config = checkpoint['config']
                    else:
                        model_config = {'input_channels': 3}
                
                # Create model instance
                self.model = DenoisingAutoencoder(**model_config)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            if hasattr(self.model, 'parameters'):
                param_count = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"Model parameters: {param_count:,}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def process_single_image(self, input_path: str, output_path: str,
                           compute_metrics: bool = False) -> Dict[str, Any]:
        """
        Process a single image through the denoising model.
        
        Args:
            input_path: Path to input image
            output_path: Path to save denoised image
            compute_metrics: Whether to compute PSNR/SSIM metrics
            
        Returns:
            Dictionary containing processing results and metrics
        """
        start_time = time.time()
        
        # Load and standardize image
        input_img = self.standardizer.load_image(input_path)
        input_tensor = self.standardizer.standardize(input_img)
        
        # Run inference
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_tensor = self.model(input_tensor)
            
            # Convert back to image
            output_img = self.standardizer.denormalize(output_tensor.cpu())
        
        # Save output image
        output_img_pil = Image.fromarray(output_img)
        output_img_pil.save(output_path)
        
        processing_time = time.time() - start_time
        
        result = {
            'input_path': input_path,
            'output_path': output_path,
            'processing_time': processing_time,
            'input_shape': input_img.shape,
            'output_shape': output_img.shape
        }
        
        # Compute metrics if requested
        if compute_metrics:
            metrics = {}
            try:
                metrics['psnr'] = calculate_psnr(input_img, output_img)
                if input_img.shape[-1] == 3:  # Color image
                    metrics['ssim'] = calculate_ssim(input_img, output_img)
            except Exception as e:
                self.logger.warning(f"Could not compute metrics: {str(e)}")
            
            result['metrics'] = metrics
        
        self.logger.info(f"Processed {input_path} in {processing_time:.3f}s")
        return result
    
    def process_folder(self, input_folder: str, output_folder: str,
                      recursive: bool = True, compute_metrics: bool = False,
                      batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Process all images in a folder through the denoising model.
        
        Args:
            input_folder: Path to folder containing input images
            output_folder: Path to save denoised images
            recursive: Whether to search recursively
            compute_metrics: Whether to compute PSNR/SSIM metrics
            batch_size: Batch size for processing
            
        Returns:
            List of processing results
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        processor = BatchProcessor(self.standardizer)
        image_files = []
        
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm', '.dicom']:
            if recursive:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
            else:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning(f"No supported image files found in {input_folder}")
            return []
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        total_time = 0
        
        for i, image_file in enumerate(image_files):
            try:
                # Determine output path
                relative_path = image_file.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Process image
                result = self.process_single_image(
                    str(image_file), str(output_file), compute_metrics
                )
                results.append(result)
                total_time += result['processing_time']
                
                # Progress logging
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    self.logger.info(f"Progress: {i + 1}/{len(image_files)} "
                                   f"({(i + 1) / len(image_files) * 100:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Summary
        avg_time = total_time / len(results) if results else 0
        self.logger.info(f"Batch processing complete:")
        self.logger.info(f"  Processed: {len(results)}/{len(image_files)} images")
        self.logger.info(f"  Total time: {total_time:.3f}s")
        self.logger.info(f"  Average time per image: {avg_time:.3f}s")
        
        return results
    
    def generate_comparison(self, input_path: str, output_path: str,
                          comparison_path: str, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Generate a side-by-side comparison of input and denoised images.
        
        Args:
            input_path: Path to input image
            output_path: Path to denoised image
            comparison_path: Path to save comparison image
            figsize: Figure size for the comparison
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Load images
            input_img = Image.open(input_path)
            output_img = Image.open(output_path)
            
            # Calculate difference
            input_array = np.array(input_img).astype(np.float32)
            output_array = np.array(output_img).astype(np.float32)
            diff_array = np.abs(input_array - output_array)
            diff_array = (diff_array - diff_array.min()) / (diff_array.max() - diff_array.min()) * 255
            diff_img = Image.fromarray(diff_array.astype(np.uint8))
            
            # Display images
            axes[0].imshow(input_img)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(output_img)
            axes[1].set_title('Denoised')
            axes[1].axis('off')
            
            axes[2].imshow(diff_img)
            axes[2].set_title('Difference')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison saved to {comparison_path}")
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping comparison generation")
        except Exception as e:
            self.logger.error(f"Error generating comparison: {str(e)}")


def main():
    """Main CLI entry point."""
    # Check if we can show help without dependencies
    if '--help' in sys.argv or '-h' in sys.argv:
        # Show help regardless of dependencies
        pass
    elif missing_deps:
        print("❌ Cannot run inference: missing required dependencies")
        print(f"Missing: {', '.join(missing_deps)}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements-ml.txt")
        print("\nShowing help anyway:")
        sys.argv.append('--help')
    
    parser = argparse.ArgumentParser(
        description="Denoising Autoencoder Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python inference_denoise.py --input test.png --output denoised.png --checkpoint model.pth
  
  # Process entire folder
  python inference_denoise.py --input-folder images/ --output-folder denoised/ --checkpoint model.pth
  
  # Process with metrics and comparison
  python inference_denoise.py --input test.png --output denoised.png --metrics --comparison --checkpoint model.pth
  
  # Process with custom image size
  python inference_denoise.py --input test.png --output denoised.png --image-size 512 512 --checkpoint model.pth
        """
    )
    
    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to input image file')
    input_group.add_argument('--input-folder', type=str, help='Path to folder containing images')
    
    parser.add_argument('--output', type=str, help='Path to output denoised image (for single image mode)')
    parser.add_argument('--output-folder', type=str, help='Path to output folder (for batch mode)')
    parser.add_argument('--comparison', action='store_true', help='Generate input/output comparison images')
    parser.add_argument('--metrics', action='store_true', help='Compute PSNR/SSIM metrics')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to model configuration file (JSON)')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='Target image size (width height)')
    
    # Processing arguments
    parser.add_argument('--recursive', action='store_true', help='Process folders recursively')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Compute device to use')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Base output directory for results')
    
    # Logging arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-error output')
    
    args = parser.parse_args()
    
    # Setup CLI
    cli = InferenceCLI()
    
    # Setup logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validate inputs
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_folder and not args.output_folder:
        parser.error("--output-folder is required when using --input-folder")
    
    if args.device != 'auto':
        if args.device == 'cpu':
            cli.device = torch.device('cpu')
        elif args.device == 'cuda' and not torch.cuda.is_available():
            cli.logger.error("CUDA not available, falling back to CPU")
            cli.device = torch.device('cpu')
    
    # Load model configuration
    model_config = None
    if args.config:
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    # Load model
    cli.logger.info(f"Loading model from {args.checkpoint}")
    cli.load_model(args.checkpoint, model_config)
    
    # Setup image standardizer
    target_size = tuple(args.image_size) if args.image_size else None
    cli.standardizer = ImageStandardizer(target_size)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        if args.input:
            # Single image mode
            input_path = Path(args.input)
            if not input_path.exists():
                cli.logger.error(f"Input file not found: {input_path}")
                return 1
            
            if not args.output:
                # Auto-generate output path
                output_path = output_dir / f"denoised_{input_path.stem}.png"
            else:
                output_path = Path(args.output)
            
            # Process image
            result = cli.process_single_image(str(input_path), str(output_path), args.metrics)
            
            # Generate comparison if requested
            if args.comparison:
                comparison_path = output_dir / f"comparison_{input_path.stem}.png"
                cli.generate_comparison(str(input_path), str(output_path), str(comparison_path))
        
        else:
            # Batch folder mode
            input_folder = Path(args.input_folder)
            output_folder = Path(args.output_folder)
            
            if not input_folder.exists():
                cli.logger.error(f"Input folder not found: {input_folder}")
                return 1
            
            # Process folder
            results = cli.process_folder(str(input_folder), str(output_folder),
                                       args.recursive, args.metrics, args.batch_size)
            
            # Save results summary
            summary_path = output_dir / "batch_summary.json"
            summary = {
                'input_folder': str(input_folder),
                'output_folder': str(output_folder),
                'total_images': len(results),
                'results': results
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            cli.logger.info(f"Batch summary saved to {summary_path}")
        
        cli.logger.info("Inference completed successfully")
        return 0
        
    except KeyboardInterrupt:
        cli.logger.info("Interrupted by user")
        return 1
    except Exception as e:
        cli.logger.error(f"Error during inference: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())