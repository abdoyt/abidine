#!/usr/bin/env python3
"""
Training script for coronary image denoising.

This script trains a lightweight CNN autoencoder to denoise low-dose coronary images.
Supports PNG, JPEG, and DICOM formats with on-the-fly noise simulation.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Import local modules
from ml.data.dataset import create_dataloaders
from ml.models.autoencoder import create_model


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image tensor
        target: Target image tensor
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM) - simplified version.
    
    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor (B, C, H, W)
        
    Returns:
        SSIM value (0 to 1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy and compute SSIM for first image in batch
        pred_np = pred[0, 0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy()
        
        ssim_val = ssim(pred_np, target_np, data_range=1.0)
        return float(ssim_val)
    except ImportError:
        # Fallback: use a simple correlation-based metric
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        correlation = (pred_centered * target_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
        
        ssim_approx = (correlation / (pred_std * target_std + 1e-8)).mean()
        return ssim_approx.item()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for noisy, clean in pbar:
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            psnr = calculate_psnr(output, clean)
            ssim = calculate_ssim(output, clean)
        
        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.2f}',
            'ssim': f'{ssim:.3f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(val_loader, desc="Validating"):
            # Move to device
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            loss = criterion(output, clean)
            
            # Calculate metrics
            psnr = calculate_psnr(output, clean)
            ssim = calculate_ssim(output, clean)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Training/validation metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_str: str) -> torch.device:
    """
    Setup compute device.
    
    Args:
        device_str: Device string ("auto", "cuda", "cpu")
        
    Returns:
        PyTorch device
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def setup_logging(log_dir: str, csv_path: str) -> Tuple[SummaryWriter, csv.DictWriter]:
    """
    Setup TensorBoard and CSV logging.
    
    Args:
        log_dir: Directory for TensorBoard logs
        csv_path: Path for CSV log file
        
    Returns:
        Tuple of (TensorBoard writer, CSV writer)
    """
    # Setup TensorBoard
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Setup CSV logging
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=['epoch', 'train_loss', 'train_psnr', 'train_ssim',
                   'val_loss', 'val_psnr', 'val_ssim']
    )
    csv_writer.writeheader()
    
    return writer, csv_writer


def train(config: dict, args: argparse.Namespace):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    # Override config with command-line arguments if provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.image_size is not None:
        config['data']['image_size'] = args.image_size
    if args.device is not None:
        config['training']['device'] = args.device
    
    # Setup device
    device = setup_device(config['training']['device'])
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_dir=config['data']['train_dir'],
        val_dir=config['data'].get('val_dir'),
        batch_size=config['data']['batch_size'],
        image_size=config['data']['image_size'],
        poisson_scale=config['noise']['poisson_scale'],
        gaussian_std=config['noise']['gaussian_std'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Setup loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Setup logging
    checkpoint_dir = config['output']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tb_writer, csv_writer = setup_logging(
        config['output']['log_dir'],
        config['output']['csv_log']
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"\nTraining   - Loss: {train_metrics['loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f} dB, "
              f"SSIM: {train_metrics['ssim']:.3f}")
        
        # Validate
        val_metrics = {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f} dB, "
                  f"SSIM: {val_metrics['ssim']:.3f}")
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
        
        # Log to TensorBoard
        tb_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        tb_writer.add_scalar('PSNR/train', train_metrics['psnr'], epoch)
        tb_writer.add_scalar('SSIM/train', train_metrics['ssim'], epoch)
        
        if val_loader is not None:
            tb_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            tb_writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
            tb_writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
        
        tb_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log to CSV
        csv_writer.writerow({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_psnr': train_metrics['psnr'],
            'train_ssim': train_metrics['ssim'],
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim']
        })
        
        # Save checkpoint
        save_every = config['output'].get('save_every', 5)
        if epoch % save_every == 0 or epoch == config['training']['epochs']:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            save_checkpoint(model, optimizer, epoch, train_metrics, checkpoint_path)
        
        # Save best model
        current_loss = val_metrics['loss'] if val_loader is not None else train_metrics['loss']
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            best_path = os.path.join(
                checkpoint_dir,
                config['output']['checkpoint_name']
            )
            save_checkpoint(model, optimizer, epoch, val_metrics or train_metrics, best_path)
            print(f"Best model saved with loss: {best_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {os.path.join(checkpoint_dir, config['output']['checkpoint_name'])}")
    print("="*60)
    
    tb_writer.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a denoising autoencoder for coronary images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/denoise_example.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        choices=[256, 512],
        default=None,
        help='Image size (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default=None,
        help='Device to use for training (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Print configuration
    print("="*60)
    print("Denoising Autoencoder Training")
    print("="*60)
    print(f"Configuration: {args.config}")
    print("="*60)
    
    # Start training
    try:
        train(config, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
