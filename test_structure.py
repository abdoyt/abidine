#!/usr/bin/env python3
"""
Basic structural test for the inference CLI without external dependencies.
Tests import structure and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test basic import structure."""
    print("Testing import structure...")
    
    # Test basic Python structure
    try:
        import argparse
        import json
        import logging
        import time
        from pathlib import Path
        from typing import Optional, Tuple, List, Dict, Any
        
        print("✓ Standard library imports work")
        
    except ImportError as e:
        print(f"✗ Standard library import failed: {e}")
        return False
    
    # Test our modules exist
    try:
        # Check if inference file exists
        inference_file = Path("inference_denoise.py")
        assert inference_file.exists(), "inference_denoise.py not found"
        print("✓ Main inference script exists")
        
        # Check ML directory structure
        ml_dir = Path("ml")
        assert ml_dir.exists(), "ml directory not found"
        
        data_dir = ml_dir / "data"
        utils_dir = ml_dir / "utils"
        assert data_dir.exists(), "ml/data directory not found"
        assert utils_dir.exists(), "ml/utils directory not found"
        print("✓ ML directory structure correct")
        
        # Check module files
        required_files = [
            "ml/__init__.py",
            "ml/data/__init__.py", 
            "ml/data/preprocessing.py",
            "ml/utils/__init__.py",
            "ml/utils/metrics.py",
            "ml/utils/model_utils.py"
        ]
        
        for file_path in required_files:
            full_path = Path(file_path)
            assert full_path.exists(), f"Required file missing: {file_path}"
        
        print("✓ All required module files exist")
        
    except AssertionError as e:
        print(f"✗ File structure test failed: {e}")
        return False
    
    return True


def test_cli_help():
    """Test CLI help functionality."""
    print("\nTesting CLI help...")
    
    try:
        # Test help message without running inference
        import subprocess
        
        # Run with --help to test argument parsing
        result = subprocess.run([
            sys.executable, "inference_denoise.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ CLI help works")
            print("Help output preview:")
            help_lines = result.stdout.split('\n')[:5]
            for line in help_lines:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"✗ CLI help failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ CLI help timed out")
        return False
    except Exception as e:
        print(f"✗ CLI help test error: {e}")
        return False


def test_basic_structure():
    """Test basic code structure without external deps."""
    print("\nTesting basic code structure...")
    
    try:
        # Read and check main script structure
        with open("inference_denoise.py", "r") as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "class InferenceCLI",
            "def main(",
            "argparse.ArgumentParser",
            "def process_single_image",
            "def process_folder",
            "def load_model"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found component: {component}")
            else:
                print(f"✗ Missing component: {component}")
                return False
        
        # Check preprocessing module
        with open("ml/data/preprocessing.py", "r") as f:
            prep_content = f.read()
        
        prep_components = [
            "class ImageStandardizer",
            "def load_image",
            "def standardize", 
            "def denormalize"
        ]
        
        for component in prep_components:
            if component in prep_content:
                print(f"✓ Found preprocessing component: {component}")
            else:
                print(f"✗ Missing preprocessing component: {component}")
                return False
        
        # Check metrics module
        with open("ml/utils/metrics.py", "r") as f:
            metrics_content = f.read()
        
        metrics_components = [
            "def calculate_psnr",
            "def calculate_ssim",
            "def evaluate_denoising"
        ]
        
        for component in metrics_components:
            if component in metrics_content:
                print(f"✓ Found metrics component: {component}")
            else:
                print(f"✗ Missing metrics component: {component}")
                return False
        
        print("✓ All code structure checks passed")
        return True
        
    except Exception as e:
        print(f"✗ Code structure test failed: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    try:
        readme_files = ["README.md", "README_INFERENCE.md", "requirements-ml.txt"]
        
        for readme_file in readme_files:
            file_path = Path(readme_file)
            if file_path.exists():
                print(f"✓ Documentation file exists: {readme_file}")
                
                # Check content isn't empty
                content = file_path.read_text()
                if len(content) > 100:
                    print(f"✓ {readme_file} has substantial content")
                else:
                    print(f"⚠ {readme_file} seems very short")
            else:
                print(f"✗ Missing documentation: {readme_file}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False


def main():
    """Run all structural tests."""
    print("Running Inference CLI Structural Tests")
    print("=" * 50)
    
    tests = [
        ("Import Structure", test_imports),
        ("CLI Help", test_cli_help), 
        ("Code Structure", test_basic_structure),
        ("Documentation", test_documentation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
        
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"Structural Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structural tests passed!")
        print("\nThe inference CLI is properly structured and ready for use.")
        print("To fully test functionality, install dependencies:")
        print("pip install -r requirements-ml.txt")
        print("\nThen run:")
        print("python test_inference.py")
    else:
        print("❌ Some structural tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)