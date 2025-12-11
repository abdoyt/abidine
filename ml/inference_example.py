"""
Example script for using a trained denoising model for inference.

This demonstrates how to load a trained model and denoise images.
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ml.models.autoencoder import create_model
from ml.data.preprocess import normalize_image, resize_image


def load_model(checkpoint_path: str, device: str = 'cpu') -> torch.nn.Module:
    """
    Load a trained denoising model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    model = create_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


def denoise_image(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int = 256,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Denoise a single image.
    
    Args:
        model: Trained denoising model
        image: Input image array
        image_size: Size to resize image to
        device: Device to run inference on
        
    Returns:
        Denoised image array
    """
    # Preprocess
    img = normalize_image(image)
    img = resize_image(img, image_size)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    
    # Convert back to numpy
    denoised = denoised_tensor.squeeze().cpu().numpy()
    
    return denoised


def main():
    parser = argparse.ArgumentParser(description='Denoise images using trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    print("Model loaded successfully")
    
    # Load image
    print(f"Loading image from {args.input}...")
    pil_img = Image.open(args.input)
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    image = np.array(pil_img, dtype=np.float32)
    
    # Denoise
    print("Denoising...")
    denoised = denoise_image(model, image, args.image_size, args.device)
    
    # Save result
    print(f"Saving denoised image to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8), mode='L')
    denoised_pil.save(args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()
