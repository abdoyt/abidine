# Guide d'intégration IA - Coro-Plus AI

## Guide technique pour intégrer des modèles de Deep Learning

Ce document explique comment remplacer les algorithmes de traitement d'image actuels par de véritables modèles d'Intelligence Artificielle basés sur le Deep Learning.

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Option 1 : Backend Python + API](#option-1--backend-python--api)
3. [Option 2 : TensorFlow.js (in-browser)](#option-2--tensorflowjs-in-browser)
4. [Préparation des données](#préparation-des-données)
5. [Entraînement des modèles](#entraînement-des-modèles)
6. [Déploiement](#déploiement)

---

## Vue d'ensemble

Le système actuel utilise des algorithmes classiques (filtre bilatéral, seuillage) pour la démonstration. Pour une application clinique réelle, il faut intégrer des modèles de Deep Learning.

### Architecture recommandée

```
┌─────────────────┐
│   Frontend      │
│   (Next.js)     │
└────────┬────────┘
         │
         ├─── Option 1: API REST ───┐
         │                          │
         │                    ┌─────▼─────┐
         │                    │  Backend  │
         │                    │  Python   │
         │                    │ (FastAPI) │
         │                    └─────┬─────┘
         │                          │
         │                    ┌─────▼─────┐
         │                    │  Modèles  │
         │                    │  PyTorch  │
         │                    └───────────┘
         │
         └─── Option 2: TensorFlow.js
                    (in-browser)
```

---

## Option 1 : Backend Python + API

### Avantages
✅ Performance optimale (GPU)
✅ Flexibilité totale
✅ Support PyTorch/TensorFlow complet
✅ Prétraitement avancé possible

### Inconvénients
❌ Infrastructure serveur nécessaire
❌ Latence réseau
❌ Coûts d'hébergement

### Étape 1 : Créer le backend Python

#### Structure du projet

```
coro-plus-ai-backend/
├── app.py                    # FastAPI application
├── models/
│   ├── __init__.py
│   ├── denoising.py         # Modèle de débruitage
│   └── segmentation.py      # Modèle de segmentation
├── weights/
│   ├── denoising_best.pth
│   └── segmentation_best.pth
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── postprocessing.py
├── requirements.txt
└── README.md
```

#### Code : app.py (FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import base64

from models.denoising import DenoisingModel
from models.segmentation import SegmentationModel
from utils.preprocessing import preprocess_image
from utils.postprocessing import postprocess_image

app = FastAPI(title="Coro-Plus AI Backend")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoising_model = DenoisingModel().to(device)
denoising_model.load_state_dict(torch.load("weights/denoising_best.pth", map_location=device))
denoising_model.eval()

segmentation_model = SegmentationModel().to(device)
segmentation_model.load_state_dict(torch.load("weights/segmentation_best.pth", map_location=device))
segmentation_model.eval()

@app.post("/api/denoise")
async def denoise_image(file: UploadFile = File(...)):
    """
    Endpoint pour le débruitage d'image
    """
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        
        # Prétraitement
        input_tensor = preprocess_image(image, device)
        
        # Inférence
        with torch.no_grad():
            output_tensor = denoising_model(input_tensor)
        
        # Post-traitement
        output_image = postprocess_image(output_tensor)
        
        # Convertir en base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Calculer les métriques
        metrics = calculate_metrics(input_tensor, output_tensor)
        
        return JSONResponse({
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "metrics": metrics
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    """
    Endpoint pour la segmentation
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        
        input_tensor = preprocess_image(image, device)
        
        with torch.no_grad():
            output_tensor = segmentation_model(input_tensor)
            output_tensor = torch.sigmoid(output_tensor)
        
        output_image = postprocess_segmentation(output_tensor)
        
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": True
    }

def calculate_metrics(input_tensor, output_tensor):
    """
    Calcule les métriques de qualité
    """
    with torch.no_grad():
        # MSE
        mse = torch.mean((input_tensor - output_tensor) ** 2).item()
        
        # PSNR
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100
        
        # SSIM (simplifiée)
        # Implémenter ou utiliser pytorch-msssim
        
    return {
        "mse": mse,
        "psnr": psnr,
        "processing_time_ms": 0  # À mesurer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

#### Code : models/denoising.py

```python
import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    """
    Autoencodeur pour le débruitage d'images médicales
    Architecture inspirée de DnCNN et RED-Net
    """
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            # Conv2
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )
        
        # Middle layers
        self.middle = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Conv4
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # Skip connection pour apprentissage résiduel
        identity = x
        
        # Encoder
        x = self.encoder(x)
        
        # Middle
        x = self.middle(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Connexion résiduelle
        x = x + identity
        
        return x

class DenoisingModel(nn.Module):
    """
    Modèle de débruitage avec skip connections
    """
    def __init__(self):
        super().__init__()
        self.model = DenoisingAutoencoder()
    
    def forward(self, x):
        return self.model(x)
```

#### Code : models/segmentation.py

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net pour segmentation des artères coronaires
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder avec skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet()
    
    def forward(self, x):
        return self.model(x)
```

#### Code : utils/preprocessing.py

```python
import torch
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, device: torch.device, target_size=512):
    """
    Prétraite l'image pour l'inférence
    """
    # Redimensionner
    image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Convertir en array numpy
    img_array = np.array(image, dtype=np.float32)
    
    # Normaliser [0, 1]
    img_array = img_array / 255.0
    
    # Ajouter dimensions batch et channel
    if len(img_array.shape) == 2:
        img_array = img_array[np.newaxis, np.newaxis, :, :]
    
    # Convertir en tensor PyTorch
    tensor = torch.from_numpy(img_array).to(device)
    
    return tensor

def add_noise_for_training(image, noise_level=0.1):
    """
    Ajoute du bruit gaussien pour l'entraînement
    """
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)
```

#### Code : utils/postprocessing.py

```python
import torch
import numpy as np
from PIL import Image

def postprocess_image(tensor: torch.Tensor):
    """
    Convertit le tensor de sortie en image PIL
    """
    # Déplacer vers CPU et enlever dimensions batch/channel
    img_array = tensor.cpu().squeeze().numpy()
    
    # Clip entre 0 et 1
    img_array = np.clip(img_array, 0, 1)
    
    # Convertir en [0, 255]
    img_array = (img_array * 255).astype(np.uint8)
    
    # Créer image PIL
    image = Image.fromarray(img_array, mode='L')
    
    return image

def postprocess_segmentation(tensor: torch.Tensor, threshold=0.5):
    """
    Post-traite la sortie de segmentation
    """
    # Appliquer seuil
    mask = (tensor > threshold).float()
    
    # Convertir en image
    img_array = mask.cpu().squeeze().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    # Créer image PIL en RGB pour visualisation
    image = Image.fromarray(img_array, mode='L')
    
    # Convertir en RGB et colorer en vert les zones segmentées
    image_rgb = Image.new('RGB', image.size)
    pixels = image.load()
    pixels_rgb = image_rgb.load()
    
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if pixels[i, j] > 128:
                pixels_rgb[i, j] = (255, 0, 0)  # Rouge pour structures
            else:
                pixels_rgb[i, j] = (pixels[i, j], pixels[i, j], pixels[i, j])
    
    return image_rgb
```

#### Code : requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
torch==2.5.0
torchvision==0.20.0
pillow==11.0.0
numpy==2.1.0
python-multipart==0.0.18
```

### Étape 2 : Modifier le frontend Next.js

#### Code : lib/api.ts (nouveau fichier)

```typescript
export async function callDenoisingAPI(imageFile: File) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:5000/api/denoise', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('API call failed');
  }
  
  const data = await response.json();
  return data;
}

export async function callSegmentationAPI(imageFile: File) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:5000/api/segment', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('API call failed');
  }
  
  const data = await response.json();
  return data;
}
```

#### Modifier app/page.tsx

```typescript
// Remplacer dans handleProcess:
if (activeTab === 'denoising') {
  // Option avec API
  const apiResult = await callDenoisingAPI(selectedFile);
  if (apiResult.success) {
    setResult({
      originalImage: URL.createObjectURL(selectedFile),
      denoisedImage: apiResult.image,
      processingTime: apiResult.metrics.processing_time_ms,
      noiseReduction: apiResult.metrics.noise_reduction || 0,
      contrastImprovement: apiResult.metrics.contrast_improvement || 0,
    });
  }
}
```

---

## Option 2 : TensorFlow.js (in-browser)

### Avantages
✅ Pas de backend nécessaire
✅ Latence minimale
✅ Pas de coûts serveur
✅ Privacy (données restent locales)

### Inconvénients
❌ Performance limitée (CPU/WebGL)
❌ Taille des modèles limitée
❌ Moins flexible

### Étape 1 : Entraîner et convertir le modèle

```python
# train_and_export.py
import tensorflow as tf
import tensorflowjs as tfjs

# Après entraînement du modèle
model.save('denoising_model')

# Convertir en TensorFlow.js
tfjs.converters.save_keras_model(model, 'public/models/denoising')
```

### Étape 2 : Utiliser dans Next.js

```bash
npm install @tensorflow/tfjs
```

```typescript
// lib/tfjs-inference.ts
import * as tf from '@tensorflow/tfjs';

let denoisingModel: tf.LayersModel | null = null;

export async function loadDenoisingModel() {
  if (!denoisingModel) {
    denoisingModel = await tf.loadLayersModel('/models/denoising/model.json');
  }
  return denoisingModel;
}

export async function denoise TensorFlowJS(imageFile: File): Promise<string> {
  const model = await loadDenoisingModel();
  
  // Charger l'image
  const img = new Image();
  img.src = URL.createObjectURL(imageFile);
  await img.decode();
  
  // Convertir en tensor
  let tensor = tf.browser.fromPixels(img, 1);
  tensor = tensor.div(255.0);
  tensor = tensor.expandDims(0);
  
  // Inférence
  const output = model.predict(tensor) as tf.Tensor;
  
  // Post-traitement
  const outputData = await output.mul(255).squeeze().array() as number[][];
  
  // Créer image
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(img.width, img.height);
  
  for (let i = 0; i < img.height; i++) {
    for (let j = 0; j < img.width; j++) {
      const idx = (i * img.width + j) * 4;
      const value = outputData[i][j];
      imageData.data[idx] = value;
      imageData.data[idx + 1] = value;
      imageData.data[idx + 2] = value;
      imageData.data[idx + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}
```

---

## Préparation des données

### Dataset recommandés

1. **ASOCA (Automated Segmentation of Coronary Arteries)**
2. **CA-500**
3. **Données simulées avec bruit ajouté**

### Script de préparation

```python
# prepare_dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CoronaryDataset(Dataset):
    def __init__(self, root_dir, noise_level=0.1, transform=None):
        self.root_dir = root_dir
        self.noise_level = noise_level
        self.transform = transform
        self.images = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        # Image propre (target)
        clean = np.array(image, dtype=np.float32) / 255.0
        
        # Image bruitée (input)
        noise = np.random.normal(0, self.noise_level, clean.shape)
        noisy = np.clip(clean + noise, 0, 1)
        
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        
        return noisy_tensor, clean_tensor
```

---

## Entraînement des modèles

### Script d'entraînement

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.denoising import DenoisingModel
from prepare_dataset import CoronaryDataset

def train_denoising_model():
    # Paramètres
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.001
    
    # Dataset
    dataset = CoronaryDataset('data/train', noise_level=0.1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modèle
    model = DenoisingModel().to(device)
    
    # Loss et optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward
            output = model(noisy)
            loss = criterion(output, clean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Sauvegarder
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'weights/denoising_epoch_{epoch+1}.pth')
    
    # Sauvegarder le modèle final
    torch.save(model.state_dict(), 'weights/denoising_best.pth')
    print('Training complete!')

if __name__ == '__main__':
    train_denoising_model()
```

---

## Déploiement

### Option A : Déploiement local

```bash
# Terminal 1 : Backend Python
cd coro-plus-ai-backend
pip install -r requirements.txt
python app.py

# Terminal 2 : Frontend Next.js
cd coro-plus-ai
npm run dev
```

### Option B : Déploiement cloud

#### Backend sur Render/Railway

```yaml
# render.yaml
services:
  - type: web
    name: coro-plus-ai-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
```

#### Frontend sur Vercel

```bash
npm run build
vercel deploy
```

---

## Conclusion

Ce guide fournit deux approches complètes pour intégrer de l'IA réelle dans Coro-Plus AI. Choisissez l'option qui correspond le mieux à vos contraintes de temps, budget et infrastructure.

Pour un prototype académique rapide → Option 2 (TensorFlow.js)
Pour une application performante → Option 1 (Backend Python)
