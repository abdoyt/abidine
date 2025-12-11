"""
Training script for coronary artery segmentation using U-Net.

This module is OPTIONAL and scaffolded for future development.
It mirrors the denoising trainer pattern with argparse configuration,
dice/bce loss, and checkpoint management.

Usage:
    python ml/train_segment.py --image-dir ./data/images --mask-dir ./data/masks
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from ml.models.unet import create_unet
from ml.data.mask_dataset import MaskDataset, create_synthetic_mask_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice coefficient loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Compute Dice loss.

        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Target masks (B, H, W)
        """
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred_flat = pred.reshape(-1)
        target_flat = target_one_hot.reshape(-1)

        intersection = torch.sum(pred_flat * target_flat)
        union = torch.sum(pred_flat) + torch.sum(target_flat)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class SegmentationLoss(nn.Module):
    """Combined Dice and BCE loss for segmentation."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train U-Net for coronary artery segmentation"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing segmentation masks",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Generate and use synthetic dataset (default: True)",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=50,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/segment",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Use data augmentation",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="Use weighted sampling for class balance",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model and optimizer state from checkpoint."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch


def save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics):
    """Save model and optimizer state to checkpoint."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Main training loop."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log config
    logger.info("=== Segmentation Training Config ===")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    # Prepare dataset
    if args.synthetic or args.image_dir is None:
        logger.info("Using synthetic dataset for validation/testing")
        img_dir, mask_dir = create_synthetic_mask_dataset(
            output_dir=str(output_dir / "synthetic_data"),
            num_samples=args.num_synthetic,
        )
    else:
        img_dir = args.image_dir
        mask_dir = args.mask_dir

    # Create datasets
    train_dataset = MaskDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        augment=args.augment,
        balance=args.balance,
    )

    # Create dataloader
    if args.balance:
        sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

    # Initialize model
    device = torch.device(args.device)
    model = create_unet(in_channels=1, out_channels=2).to(device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Setup training
    criterion = SegmentationLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)

        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                str(output_dir / "best_model.pth"),
                model,
                optimizer,
                epoch,
                {"loss": train_loss},
            )

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                str(output_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"),
                model,
                optimizer,
                epoch,
                {"loss": train_loss},
            )

    # Save final model
    save_checkpoint(
        str(output_dir / "final_model.pth"),
        model,
        optimizer,
        args.epochs,
        {"loss": train_loss},
    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Training completed. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
