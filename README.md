This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## ML Package: Image Denoising

This project includes a complete machine learning pipeline for training denoising autoencoders on coronary medical images.

### Quick Start - ML Training

1. **Install ML dependencies**:
```bash
pip install -r requirements-ml.txt
```

2. **Generate test data** (or use your own images):
```bash
python ml/generate_test_data.py --output-dir data/train --num-images 50 --val
```

3. **Train the model**:
```bash
python train_denoise.py --config configs/denoise_example.yaml
```

4. **Monitor training**:
```bash
tensorboard --logdir artifacts/logs
```

### Features

- **Multi-format support**: PNG, JPEG, DICOM
- **On-the-fly noise simulation**: Poisson + Gaussian noise for low-dose imaging
- **Lightweight U-Net autoencoder**: ~1.2M parameters
- **Comprehensive logging**: TensorBoard + CSV with PSNR/SSIM metrics
- **GPU/CPU support**: Automatic device selection
- **Checkpointing**: Automatic saving of best models

See [ml/README.md](ml/README.md) for detailed documentation.

## Getting Started - Next.js

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

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
