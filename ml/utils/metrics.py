"""
Metrics calculation utilities for image quality assessment.
Includes PSNR and SSIM calculations for denoising evaluation.
"""

import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, 
                  data_range: Optional[float] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (ground truth)
        img2: Second image (prediction)
        data_range: Data range of the images. If None, inferred from img1.
        
    Returns:
        PSNR value in dB
    """
    # Ensure images are numpy arrays
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    # Determine data range if not provided
    if data_range is None:
        data_range = img1.max() - img1.min()
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Handle edge case
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, 
                  data_range: Optional[float] = None) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1: First image (ground truth)
        img2: Second image (prediction)
        data_range: Data range of the images. If None, inferred from img1.
        
    Returns:
        SSIM value (0-1, higher is better)
    """
    # Convert to grayscale for SSIM calculation if color images
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        # Convert RGB to grayscale
        img1_gray = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
        img2_gray = np.dot(img2[...,:3], [0.299, 0.587, 0.114])
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Ensure proper data types
    img1_gray = np.asarray(img1_gray, dtype=np.float64)
    img2_gray = np.asarray(img2_gray, dtype=np.float64)
    
    # Determine data range if not provided
    if data_range is None:
        data_range = img1.max() - img1.min()
    
    # Use scikit-image SSIM
    try:
        # Handle different data ranges
        if data_range > 1:
            # Assume 0-255 range
            img1_norm = img1_gray / 255.0
            img2_norm = img2_gray / 255.0
        else:
            # Assume 0-1 range
            img1_norm = img1_gray
            img2_norm = img2_gray
        
        ssim_value = ssim(img1_norm, img2_norm, 
                         data_range=1.0, 
                         gaussian_weights=True, 
                         sigma=1.5,
                         use_sample_covariance=False)
        
        return float(ssim_value)
    
    except Exception as e:
        # Fallback to simple implementation
        return _calculate_ssim_fallback(img1_gray, img2_gray)


def _calculate_ssim_fallback(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Fallback SSIM implementation if scikit-image is not available.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        SSIM value
    """
    # Constants for SSIM calculation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_value = numerator / denominator
    return float(np.clip(ssim_value, -1, 1))


def calculate_batch_psnr(imgs1: list, imgs2: list, 
                        data_range: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate PSNR for a batch of images.
    
    Args:
        imgs1: List of first images
        imgs2: List of second images
        data_range: Data range of the images
        
    Returns:
        Tuple of (mean_psnr, std_psnr)
    """
    psnr_values = []
    
    for img1, img2 in zip(imgs1, imgs2):
        psnr = calculate_psnr(img1, img2, data_range)
        psnr_values.append(psnr)
    
    return float(np.mean(psnr_values)), float(np.std(psnr_values))


def calculate_batch_ssim(imgs1: list, imgs2: list, 
                        data_range: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate SSIM for a batch of images.
    
    Args:
        imgs1: List of first images
        imgs2: List of second images
        data_range: Data range of the images
        
    Returns:
        Tuple of (mean_ssim, std_ssim)
    """
    ssim_values = []
    
    for img1, img2 in zip(imgs1, imgs2):
        ssim_val = calculate_ssim(img1, img2, data_range)
        ssim_values.append(ssim_val)
    
    return float(np.mean(ssim_values)), float(np.std(ssim_values))


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MSE value
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    return float(mse)


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MAE value
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    mae = np.mean(np.abs(img1 - img2))
    return float(mae)


def evaluate_denoising(original: np.ndarray, denoised: np.ndarray, 
                      reference: Optional[np.ndarray] = None) -> dict:
    """
    Comprehensive evaluation of denoising performance.
    
    Args:
        original: Original noisy image
        denoised: Denoised image
        reference: Optional reference clean image for ground truth metrics
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Always calculate metrics between original and denoised
    metrics['psnr_original_denoised'] = calculate_psnr(original, denoised)
    metrics['ssim_original_denoised'] = calculate_ssim(original, denoised)
    metrics['mse_original_denoised'] = calculate_mse(original, denoised)
    metrics['mae_original_denoised'] = calculate_mae(original, denoised)
    
    # If reference is available, calculate additional metrics
    if reference is not None:
        metrics['psnr_reference_denoised'] = calculate_psnr(reference, denoised)
        metrics['ssim_reference_denoised'] = calculate_ssim(reference, denoised)
        metrics['mse_reference_denoised'] = calculate_mse(reference, denoised)
        metrics['mae_reference_denoised'] = calculate_mae(reference, denoised)
        
        # Improvement metrics
        psnr_improvement = (metrics['psnr_reference_denoised'] - 
                          calculate_psnr(reference, original))
        metrics['psnr_improvement'] = psnr_improvement
        
        ssim_improvement = (metrics['ssim_reference_denoised'] - 
                          calculate_ssim(reference, original))
        metrics['ssim_improvement'] = ssim_improvement
    
    return metrics


class MetricsCalculator:
    """Utility class for batch metrics calculation."""
    
    def __init__(self, data_range: Optional[float] = None):
        """
        Initialize metrics calculator.
        
        Args:
            data_range: Data range of images (e.g., 255 for 8-bit, 1.0 for normalized)
        """
        self.data_range = data_range
    
    def evaluate_folder(self, original_folder: str, denoised_folder: str,
                       reference_folder: Optional[str] = None) -> dict:
        """
        Evaluate all images in folders.
        
        Args:
            original_folder: Folder containing original noisy images
            denoised_folder: Folder containing denoised images
            reference_folder: Optional folder containing reference clean images
            
        Returns:
            Dictionary containing aggregated metrics
        """
        from pathlib import Path
        
        original_path = Path(original_folder)
        denoised_path = Path(denoised_folder)
        
        if not original_path.exists():
            raise FileNotFoundError(f"Original folder not found: {original_folder}")
        if not denoised_path.exists():
            raise FileNotFoundError(f"Denoised folder not found: {denoised_folder}")
        
        reference_path = None
        if reference_folder:
            reference_path = Path(reference_folder)
            if not reference_path.exists():
                raise FileNotFoundError(f"Reference folder not found: {reference_folder}")
        
        # Get all image files (assuming same filenames in all folders)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            for img_file in original_path.glob(f"*{ext}"):
                image_files.append(img_file)
            for img_file in original_path.glob(f"*{ext.upper()}"):
                image_files.append(img_file)
        
        all_metrics = []
        
        for img_file in image_files:
            try:
                # Load images
                original_img = self._load_image(original_path / img_file.name)
                denoised_img = self._load_image(denoised_path / img_file.name)
                
                reference_img = None
                if reference_path and (reference_path / img_file.name).exists():
                    reference_img = self._load_image(reference_path / img_file.name)
                
                # Calculate metrics
                metrics = evaluate_denoising(original_img, denoised_img, reference_img)
                metrics['filename'] = img_file.name
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"Warning: Could not process {img_file.name}: {str(e)}")
                continue
        
        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)
        aggregated['num_images'] = len(all_metrics)
        aggregated['individual_results'] = all_metrics
        
        return aggregated
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from path."""
        from PIL import Image
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def _aggregate_metrics(self, metrics_list: list) -> dict:
        """Aggregate metrics from multiple images."""
        if not metrics_list:
            return {}
        
        # Get all metric keys
        metric_keys = [key for key in metrics_list[0].keys() 
                      if key not in ['filename']]
        
        aggregated = {}
        
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated