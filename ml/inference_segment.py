"""
Inference script for coronary artery segmentation.

This module is OPTIONAL and scaffolded for future development.
Consumes denoised images and writes colorized artery overlays
suitable for display in the Streamlit app.

Usage:
    python ml/inference_segment.py --model ./checkpoints/segment/best_model.pth \\
                                   --input-dir ./data/denoised \\
                                   --output-dir ./data/segmented
"""

import argparse
import logging
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from ml.models.unet import create_unet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SegmentationInference:
    """Handles segmentation inference and colorization."""

    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = create_unet(in_channels=1, out_channels=2).to(self.device)

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state"])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(
                f"Model not found at {model_path}, using random initialization"
            )

        self.model.eval()

    def load_image(self, image_path):
        """Load and normalize image."""
        if str(image_path).endswith(".npy"):
            image = np.load(image_path)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Add channel and batch dims: (H, W) -> (1, 1, H, W)
        image = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float()
        return image

    def segment(self, image):
        """Run segmentation on image."""
        with torch.no_grad():
            image = image.to(self.device)
            logits = self.model(image)
            probs = F.softmax(logits, dim=1)
            # Get artery class probability (class 1)
            artery_prob = probs[0, 1].cpu().numpy()
        return artery_prob

    def colorize_overlay(self, image_np, artery_mask, alpha=0.5, threshold=0.5):
        """
        Create colorized overlay of artery segmentation.

        Args:
            image_np: Original image (H, W) in [0, 1] or [0, 255]
            artery_mask: Artery probability mask (H, W) in [0, 1]
            alpha: Opacity of overlay
            threshold: Confidence threshold for artery pixels

        Returns:
            Colorized overlay image as RGB uint8 array
        """
        # Ensure image is in [0, 255] uint8
        if image_np.max() <= 1:
            image_rgb = (image_np * 255).astype(np.uint8)
        else:
            image_rgb = image_np.astype(np.uint8)

        # Create RGB from grayscale
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

        # Create binary mask for confident artery predictions
        binary_mask = (artery_mask > threshold).astype(np.uint8)

        # Colorize arteries in red
        colored_overlay = image_rgb.copy()
        colored_overlay[binary_mask == 1] = [0, 0, 255]  # Red in BGR

        # Blend with original
        result = cv2.addWeighted(
            image_rgb, 1 - alpha, colored_overlay, alpha, 0
        )

        return result

    def save_outputs(self, image_path, output_dir, save_mask=True, save_overlay=True):
        """
        Run inference and save outputs.

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_mask: Whether to save probability mask
            save_overlay: Whether to save colorized overlay
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and infer
        image = self.load_image(image_path)
        image_np = image.squeeze().numpy()
        artery_prob = self.segment(image)

        base_name = Path(image_path).stem

        # Save artery probability mask
        if save_mask:
            mask_path = output_dir / f"{base_name}_mask.npy"
            np.save(mask_path, artery_prob)
            logger.info(f"Saved mask to {mask_path}")

            # Also save as image for visualization
            mask_img = (artery_prob * 255).astype(np.uint8)
            mask_img_path = output_dir / f"{base_name}_mask.png"
            Image.fromarray(mask_img).save(mask_img_path)
            logger.info(f"Saved mask image to {mask_img_path}")

        # Save colorized overlay
        if save_overlay:
            overlay = self.colorize_overlay(image_np, artery_prob)
            overlay_path = output_dir / f"{base_name}_overlay.png"
            Image.fromarray(overlay).save(overlay_path)
            logger.info(f"Saved overlay to {overlay_path}")

        return {
            "image": image_path,
            "artery_prob": artery_prob,
            "mask_saved": save_mask,
            "overlay_saved": save_overlay,
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run segmentation inference on images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/denoised",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/segmented",
        help="Directory to save segmented outputs",
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="*.npy",
        help="Pattern for input images (*.npy or *.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for artery pixels",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay opacity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    return parser.parse_args()


def main():
    """Main inference loop."""
    args = parse_args()

    logger.info("=== Segmentation Inference Config ===")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    # Initialize inference
    inference = SegmentationInference(args.model, device=args.device)

    # Process images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        logger.info("Skipping inference")
        return

    image_files = sorted(
        list(input_dir.glob("*.npy")) + list(input_dir.glob("*.png"))
    )
    logger.info(f"Found {len(image_files)} images to process")

    results = []
    for i, image_file in enumerate(image_files, 1):
        try:
            result = inference.save_outputs(
                image_file,
                args.output_dir,
                save_mask=True,
                save_overlay=True,
            )
            results.append(result)
            logger.info(f"[{i}/{len(image_files)}] Processed {image_file.name}")
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")

    # Save inference summary
    summary_path = Path(args.output_dir) / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_processed": len(results),
                "output_dir": str(args.output_dir),
                "threshold": args.threshold,
                "alpha": args.alpha,
            },
            f,
            indent=2,
        )

    logger.info(f"Inference completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
