"""
Generate synthetic test images for trying out the denoising pipeline.

This script creates simple synthetic medical-like images that can be used
to test the training pipeline without needing real medical data.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_coronary_image(size: int = 256) -> np.ndarray:
    """
    Create a synthetic coronary-like image with vessels and structures.
    
    Args:
        size: Image size (square)
        
    Returns:
        Synthetic image as numpy array
    """
    # Create base image
    img = Image.new('L', (size, size), color=20)
    draw = ImageDraw.Draw(img)
    
    # Add some vessel-like structures (curved lines)
    for i in range(3, 8):
        # Random curved vessel
        points = []
        x = np.random.randint(0, size//4)
        for step in range(20):
            y = size * step // 20
            x += np.random.randint(-10, 10)
            x = max(10, min(size-10, x))
            points.append((x, y))
        
        # Draw vessel with varying width
        for j in range(len(points)-1):
            width = np.random.randint(2, 6)
            draw.line([points[j], points[j+1]], fill=180, width=width)
    
    # Add some circular structures (vessel cross-sections)
    for _ in range(5):
        cx = np.random.randint(20, size-20)
        cy = np.random.randint(20, size-20)
        r = np.random.randint(5, 15)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=150, outline=180)
    
    # Add some noise texture
    img_array = np.array(img, dtype=np.float32)
    texture = np.random.normal(0, 10, (size, size))
    img_array = np.clip(img_array + texture, 0, 255)
    
    return img_array


def generate_dataset(output_dir: str, num_images: int = 50, size: int = 256):
    """
    Generate a synthetic dataset.
    
    Args:
        output_dir: Output directory for images
        num_images: Number of images to generate
        size: Image size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} synthetic images...")
    
    for i in range(num_images):
        # Generate image
        img_array = create_synthetic_coronary_image(size)
        
        # Save as PNG
        img_pil = Image.fromarray(img_array.astype(np.uint8), mode='L')
        output_path = os.path.join(output_dir, f'synthetic_{i:04d}.png')
        img_pil.save(output_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_images} images")
    
    print(f"Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test images for denoising training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/train',
        help='Output directory for generated images'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=50,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Image size (square)'
    )
    parser.add_argument(
        '--val',
        action='store_true',
        help='Also generate validation set (20%% of training size)'
    )
    
    args = parser.parse_args()
    
    # Generate training set
    generate_dataset(args.output_dir, args.num_images, args.size)
    
    # Generate validation set if requested
    if args.val:
        val_dir = args.output_dir.replace('/train', '/val')
        val_num = max(10, args.num_images // 5)
        print("\nGenerating validation set...")
        generate_dataset(val_dir, val_num, args.size)
    
    print("\nDone! You can now run training with:")
    print(f"  python train_denoise.py")


if __name__ == '__main__':
    main()
