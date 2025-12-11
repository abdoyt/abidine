#!/usr/bin/env python3
"""
Quick validation script to check if the ML package is set up correctly.
Run this after installing dependencies to verify everything works.
"""

import sys
import importlib


def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'pydicom',
        'skimage',
        'tensorboard',
        'yaml',
        'tqdm'
    ]
    
    print("Checking dependencies...")
    missing = []
    
    for module in required:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements-ml.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_package_structure():
    """Check if the ML package structure is correct."""
    print("\nChecking package structure...")
    
    try:
        import ml
        print("  ✓ ml")
        
        import ml.data.dataset
        print("  ✓ ml.data.dataset")
        
        import ml.data.preprocess
        print("  ✓ ml.data.preprocess")
        
        import ml.models.autoencoder
        print("  ✓ ml.models.autoencoder")
        
        print("\n✓ Package structure is correct!")
        return True
    except ImportError as e:
        print(f"\n✗ Package import failed: {e}")
        return False


def check_model_creation():
    """Test if the model can be created."""
    print("\nTesting model creation...")
    
    try:
        from ml.models.autoencoder import create_model
        import torch
        
        model = create_model(base_channels=16)  # Small model for testing
        print(f"  ✓ Model created with {model.count_parameters():,} parameters")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
        
        print("\n✓ Model works correctly!")
        return True
    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        return False


def check_preprocessing():
    """Test preprocessing functions."""
    print("\nTesting preprocessing...")
    
    try:
        import numpy as np
        from ml.data.preprocess import normalize_image, resize_image, simulate_low_dose
        
        # Create test image
        test_img = np.random.rand(512, 512).astype(np.float32) * 255
        
        # Test normalization
        normalized = normalize_image(test_img)
        assert normalized.min() >= 0 and normalized.max() <= 1
        print("  ✓ Normalization works")
        
        # Test resizing
        resized = resize_image(normalized, 256)
        assert resized.shape == (256, 256)
        print("  ✓ Resizing works")
        
        # Test noise simulation
        noisy = simulate_low_dose(normalized)
        assert noisy.shape == normalized.shape
        print("  ✓ Noise simulation works")
        
        print("\n✓ Preprocessing functions work correctly!")
        return True
    except Exception as e:
        print(f"\n✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("ML Package Validation")
    print("="*60)
    
    all_pass = True
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install dependencies first:")
        print("   pip install -r requirements-ml.txt")
        sys.exit(1)
    
    # Check package structure
    if not check_package_structure():
        all_pass = False
    
    # Check model
    if not check_model_creation():
        all_pass = False
    
    # Check preprocessing
    if not check_preprocessing():
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to train! Try:")
        print("  1. Generate test data: python ml/generate_test_data.py --val")
        print("  2. Start training: python train_denoise.py")
        print("  3. Monitor: tensorboard --logdir artifacts/logs")
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        sys.exit(1)


if __name__ == '__main__':
    main()
