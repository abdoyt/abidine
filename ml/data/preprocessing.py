"""
Preprocessing utilities for image denoising.
Supports PNG, JPEG, and DICOM file formats.
"""

from pathlib import Path
from typing import Union, Tuple, Optional

# Conditional imports for graceful degradation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic functionality")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL (Pillow) not available")

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not available, DICOM support disabled")

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not available")


class ImageStandardizer:
    """Handles standardization of various image formats for model input."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the image standardizer.
        
        Args:
            target_size: Target size for images (width, height). If None, keeps original size.
        """
        self.target_size = target_size
        
        if HAS_TORCHVISION:
            self.to_tensor = transforms.ToTensor()
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        else:
            self.to_tensor = None
            self.normalize = None
    
    def load_image(self, image_path: Union[str, Path]):
        """
        Load image from various formats (PNG, JPEG, DICOM).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array representation of the image
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not HAS_PIL:
            raise ImportError("PIL (Pillow) required for image loading")
        
        file_ext = image_path.suffix.lower()
        
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            # Handle standard image formats
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if HAS_NUMPY:
                return np.array(img)
            else:
                # Fallback for basic functionality
                return img
        
        elif file_ext in ['.dcm', '.dicom']:
            # Handle DICOM format
            return self._load_dicom(image_path)
        
        else:
            raise ValueError(f"Unsupported image format: {file_ext}")
    
    def _load_dicom(self, dicom_path: Union[str, Path]):
        """
        Load DICOM file and convert to numpy array.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            numpy array representation of the DICOM image
        """
        if not HAS_PYDICOM:
            raise ImportError("pydicom required for DICOM support")
        
        if not HAS_NUMPY:
            raise ImportError("numpy required for DICOM processing")
        
        try:
            ds = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            img_array = ds.pixel_array
            
            # Handle different pixel representations
            if img_array.dtype == np.uint16:
                # Normalize 16-bit images to 8-bit for processing
                img_array = ((img_array - img_array.min()) / 
                           (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Convert grayscale to RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            return img_array.astype(np.uint8)
        
        except Exception as e:
            raise ValueError(f"Error loading DICOM file {dicom_path}: {str(e)}")
    
    def standardize(self, image):
        """
        Standardize image for model input.
        
        Args:
            image: numpy array image
            
        Returns:
            standardized tensor ready for model input
        """
        if not HAS_NUMPY or not HAS_TORCHVISION:
            raise ImportError("numpy and torchvision required for standardization")
        
        # Resize if target size specified
        if self.target_size and HAS_PIL:
            img = Image.fromarray(image)
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            image = np.array(img)
        
        # Convert to tensor and normalize
        tensor = self.to_tensor(image)
        tensor = self.normalize(tensor)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def denormalize(self, tensor):
        """
        Convert model output tensor back to image format.
        
        Args:
            tensor: model output tensor
            
        Returns:
            numpy array image
        """
        if not HAS_NUMPY or not HAS_TORCHVISION:
            raise ImportError("numpy and torchvision required for denormalization")
        
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        
        # Convert to PIL image
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        return array


def get_supported_formats() -> list:
    """Return list of supported image formats."""
    return ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm', '.dicom']


def validate_image_file(image_path: Union[str, Path]) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if supported format, False otherwise
    """
    image_path = Path(image_path)
    supported_exts = get_supported_formats()
    return image_path.suffix.lower() in supported_exts


import torch


class BatchProcessor:
    """Handle batch processing of images."""
    
    def __init__(self, standardizer: ImageStandardizer):
        """
        Initialize batch processor.
        
        Args:
            standardizer: ImageStandardizer instance
        """
        self.standardizer = standardizer
    
    def process_folder(self, folder_path: Union[str, Path], 
                      recursive: bool = True) -> list:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            recursive: Whether to search recursively
            
        Returns:
            List of standardized image tensors
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all supported image files
        pattern = "**/*" if recursive else "*"
        image_files = []
        
        for ext in get_supported_formats():
            image_files.extend(folder_path.glob(f"{pattern}{ext}"))
            image_files.extend(folder_path.glob(f"{pattern}{ext.upper()}"))
        
        # Process each image
        processed_images = []
        for image_file in image_files:
            try:
                img_array = self.standardizer.load_image(image_file)
                tensor = self.standardizer.standardize(img_array)
                processed_images.append((image_file, tensor))
            except Exception as e:
                print(f"Warning: Could not process {image_file}: {str(e)}")
        
        return processed_images