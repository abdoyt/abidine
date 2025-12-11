#!/usr/bin/env python3
"""
Test script for the denoising inference CLI.
Creates sample images and tests both single image and batch processing modes.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from inference_denoise import InferenceCLI, DenoisingAutoencoder
    from ml.data.preprocessing import ImageStandardizer
    from ml.utils.metrics import calculate_psnr, calculate_ssim
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed:")
    print("pip install -r requirements-ml.txt")
    sys.exit(1)


def create_test_image(width: int = 256, height: int = 256, 
                     noise_level: float = 0.1) -> Image.Image:
    """Create a test image with some pattern and optional noise."""
    # Create a simple test pattern
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some patterns
    for i in range(height):
        for j in range(width):
            # Create a gradient pattern
            img_array[i, j, 0] = int((i / height) * 255)  # Red gradient
            img_array[i, j, 1] = int((j / width) * 255)  # Green gradient
            img_array[i, j, 2] = int(((i + j) / (height + width)) * 255)  # Blue pattern
    
    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def create_dicom_like_image(width: int = 256, height: int = 256) -> Image.Image:
    """Create a test image that simulates DICOM format."""
    # Create a grayscale medical-like pattern
    img_array = np.zeros((height, width), dtype=np.uint8)
    
    # Add circular pattern (like organ)
    center_x, center_y = width // 2, height // 2
    for i in range(height):
        for j in range(width):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if distance < min(width, height) // 4:
                img_array[i, j] = int(200 + 50 * np.cos(distance / 10))
            else:
                img_array[i, j] = int(50 + 30 * np.sin(distance / 15))
    
    # Convert to RGB (simulating DICOM to RGB conversion)
    img_array = np.stack([img_array] * 3, axis=-1)
    return Image.fromarray(img_array)


def test_single_image_processing():
    """Test single image processing."""
    print("Testing single image processing...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        clean_image = create_test_image(128, 128, noise_level=0.0)
        noisy_image = create_test_image(128, 128, noise_level=0.2)
        
        input_path = temp_path / "input.png"
        output_path = temp_path / "output.png"
        comparison_path = temp_path / "comparison.png"
        
        clean_image.save(input_path)
        print(f"Created test image: {input_path}")
        
        # Initialize CLI
        cli = InferenceCLI()
        
        # Create a simple model
        model = DenoisingAutoencoder()
        cli.model = model
        cli.standardizer = ImageStandardizer(target_size=(128, 128))
        
        # Process image
        result = cli.process_single_image(str(input_path), str(output_path), compute_metrics=True)
        print(f"Processing result: {result}")
        
        # Generate comparison
        cli.generate_comparison(str(input_path), str(output_path), str(comparison_path))
        
        # Verify outputs exist
        assert output_path.exists(), "Output image was not created"
        assert comparison_path.exists(), "Comparison image was not created"
        
        print("✓ Single image processing test passed")
        return True


def test_batch_processing():
    """Test batch folder processing."""
    print("Testing batch processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_folder = temp_path / "input"
        output_folder = temp_path / "output"
        
        input_folder.mkdir()
        output_folder.mkdir()
        
        # Create multiple test images
        for i in range(5):
            img = create_test_image(64, 64, noise_level=0.1)
            img.save(input_folder / f"test_{i:02d}.png")
        
        print(f"Created {len(list(input_folder.glob('*.png')))} test images")
        
        # Initialize CLI
        cli = InferenceCLI()
        
        # Create a simple model
        model = DenoisingAutoencoder()
        cli.model = model
        cli.standardizer = ImageStandardizer(target_size=(64, 64))
        
        # Process batch
        results = cli.process_folder(str(input_folder), str(output_folder), 
                                   recursive=False, compute_metrics=True)
        
        # Verify outputs
        output_files = list(output_folder.glob("*.png"))
        assert len(output_files) == 5, f"Expected 5 output files, got {len(output_files)}"
        
        print(f"✓ Batch processing test passed - processed {len(results)} images")
        return True


def test_metrics_calculation():
    """Test metrics calculation functions."""
    print("Testing metrics calculation...")
    
    # Create two test images
    img1 = np.array(create_test_image(64, 64, noise_level=0.0))
    img2 = np.array(create_test_image(64, 64, noise_level=0.0))
    
    # Calculate metrics
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"PSNR between identical images: {psnr:.2f}")
    print(f"SSIM between identical images: {ssim:.4f}")
    
    # PSNR should be very high for identical images
    assert psnr > 50, f"PSNR too low for identical images: {psnr}"
    assert ssim > 0.99, f"SSIM too low for identical images: {ssim}"
    
    # Test with different images
    img3 = np.array(create_test_image(64, 64, noise_level=0.5))
    psnr_diff = calculate_psnr(img1, img3)
    ssim_diff = calculate_ssim(img1, img3)
    
    print(f"PSNR between different images: {psnr_diff:.2f}")
    print(f"SSIM between different images: {ssim_diff:.4f}")
    
    assert psnr_diff < psnr, "PSNR should be lower for different images"
    
    print("✓ Metrics calculation test passed")
    return True


def test_supported_formats():
    """Test support for different image formats."""
    print("Testing different image formats...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create images in different formats
        formats = ['png', 'jpg', 'jpeg']
        
        for fmt in formats:
            img = create_test_image(32, 32)
            img_path = temp_path / f"test.{fmt}"
            img.save(img_path, quality=95 if fmt in ['jpg', 'jpeg'] else None)
            
            # Test loading with ImageStandardizer
            standardizer = ImageStandardizer()
            loaded_img = standardizer.load_image(img_path)
            
            assert loaded_img is not None, f"Failed to load {fmt} format"
            assert loaded_img.shape[2] == 3, f"Wrong number of channels for {fmt} format"
        
        print("✓ Image format support test passed")
        return True


def run_all_tests():
    """Run all tests."""
    print("Running inference CLI tests...")
    print("=" * 50)
    
    tests = [
        ("Single Image Processing", test_single_image_processing),
        ("Batch Processing", test_batch_processing),
        ("Metrics Calculation", test_metrics_calculation),
        ("Supported Formats", test_supported_formats),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


def create_example_scripts():
    """Create example usage scripts."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Single image example
    single_image_example = """#!/usr/bin/env python3
\"\"\"
Example script for single image denoising.
\"\"\"

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_denoise import main

if __name__ == "__main__":
    # Example: Process a single image
    import argparse
    
    # Override sys.argv for demo
    sys.argv = [
        "inference_denoise.py",
        "--input", "examples/sample_input.png",
        "--output", "examples/sample_output.png", 
        "--checkpoint", "examples/model_checkpoint.pth",
        "--metrics",
        "--comparison"
    ]
    
    main()
"""
    
    # Batch processing example
    batch_example = """#!/usr/bin/env python3
\"\"\"
Example script for batch folder denoising.
\"\"\"

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_denoise import main

if __name__ == "__main__":
    # Example: Process entire folder
    import argparse
    
    # Override sys.argv for demo
    sys.argv = [
        "inference_denoise.py",
        "--input-folder", "examples/input_folder/",
        "--output-folder", "examples/output_folder/",
        "--checkpoint", "examples/model_checkpoint.pth",
        "--recursive",
        "--metrics",
        "--batch-size", "4"
    ]
    
    main()
"""
    
    # Write example files
    (examples_dir / "single_image_example.py").write_text(single_image_example)
    (examples_dir / "batch_example.py").write_text(batch_example)
    
    print(f"Example scripts created in {examples_dir}")


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        create_example_scripts()
        print("\n🚀 All systems ready for inference!")
        print("\nUsage examples:")
        print("  # Single image")
        print("  python inference_denoise.py --input image.png --output denoised.png --checkpoint model.pth")
        print("")
        print("  # Batch processing")
        print("  python inference_denoise.py --input-folder images/ --output-folder denoised/ --checkpoint model.pth")
        print("")
        print("  # With metrics and comparison")
        print("  python inference_denoise.py --input image.png --output denoised.png --metrics --comparison --checkpoint model.pth")
    
    sys.exit(0 if success else 1)