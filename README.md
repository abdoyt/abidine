This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## ML Modules

### Module B: Segmentation (OPTIONAL)

The segmentation module is scaffolded and ready for development. It includes:

#### Components

- **U-Net Model** (`ml/models/unet.py`): Lightweight 2D U-Net architecture for coronary artery segmentation
- **Mask Dataset** (`ml/data/mask_dataset.py`): Dataset utilities for pairing images with masks, including:
  - Support for .npy and .png files
  - Data augmentation (rotation, flip, brightness/contrast adjustments)
  - Class-balancing via weighted sampling
  - Synthetic dataset generation for validation
- **Training** (`ml/train_segment.py`): Training script with:
  - Argparse configuration
  - Dice + BCE loss
  - Model checkpointing
  - LR scheduling
  - Support for synthetic datasets
- **Inference** (`ml/inference_segment.py`): Inference script that:
  - Consumes denoised images
  - Generates artery probability masks
  - Writes colorized overlays for Streamlit visualization

#### Usage

Generate synthetic dataset and train:

```bash
python ml/train_segment.py --synthetic --num-synthetic 50 --epochs 20
```

Run inference on denoised images:

```bash
python ml/inference_segment.py \
  --model ./checkpoints/segment/best_model.pth \
  --input-dir ./data/denoised \
  --output-dir ./data/segmented
```

Create synthetic dataset for testing:

```python
from ml.data.mask_dataset import create_synthetic_mask_dataset
img_dir, mask_dir = create_synthetic_mask_dataset(output_dir="./data/synthetic")
```

#### Notes

- Module is optional and can be skipped in production
- All scripts work with synthetic or small public mask datasets
- Colorized overlays are designed for integration with Streamlit app
- Probability masks are saved as both .npy (for processing) and .png (for visualization)

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
