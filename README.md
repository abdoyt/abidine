# Coro-Plus AI

## Système IA pour l'amélioration du coroscanner en imagerie coronaire

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-MVP-green)
![License](https://img.shields.io/badge/license-Academic-orange)

### 📋 Présentation du projet

**Coro-Plus AI** est un prototype académique d'Intelligence Artificielle conçu pour améliorer les images de coroscanner en imagerie coronaire. Développé dans le cadre d'un projet de Licence Professionnelle 3ème année en Manipulateur en Imagerie Médicale à l'INFSPM Oran par **Abidine**.

#### Contexte médical

Le coroscanner (CT coronaire) est un examen très performant pour l'anatomie coronaire mais présente plusieurs inconvénients :
- Dose de rayons X élevée
- Nécessité de produit de contraste iodé
- Présence de bruit et d'artéfacts dans les images
- Temps de post-traitement important
- Manque d'informations fonctionnelles

**Coro-Plus AI** vise à améliorer ces aspects grâce à l'Intelligence Artificielle.

---

## 🎯 Objectifs du système

Le prototype offre deux modules principaux :

### Module A - Débruitage et amélioration d'image (Prioritaire)
- Réduction du bruit dans les images de coroscanner
- Amélioration du contraste pour une meilleure lisibilité
- Préservation des détails anatomiques importants
- Potentiel de réduction de dose de rayonnement

### Module B - Segmentation coronaire (Démonstratif)
- Détection basique des structures vasculaires
- Mise en évidence des artères coronaires principales
- Base pour analyse quantitative future

### Module C - Génération de rapport
- Rapport automatique avec métriques quantitatives
- Temps de traitement
- Pourcentage de réduction du bruit
- Amélioration du contraste

---

## 🚀 Installation et utilisation

### Prérequis

- Node.js 20+ installé
- npm ou pnpm

### Installation

```bash
# Cloner le dépôt
git clone <repository-url>
cd coro-plus-ai

# Installer les dépendances
npm install

# Lancer le serveur de développement
npm run dev
```

### Utilisation

1. Ouvrir le navigateur à `http://localhost:3000`
2. Sélectionner un module (Débruitage ou Segmentation)
3. Charger une image de coroscanner (PNG, JPEG)
4. Cliquer sur "Appliquer le traitement"
5. Visualiser les résultats avant/après
6. Télécharger les images traitées
7. Générer un rapport d'analyse

### Format des images

- **Format supporté actuellement** : PNG, JPEG
- **Format DICOM** : Convertir en PNG/JPEG avant utilisation
- **Résolution recommandée** : 512×512 ou 256×256 pixels
- **Type d'image** : Images de coroscanner en niveaux de gris ou couleur

---

## 🏗️ Architecture technique

### Stack technologique

- **Frontend** : Next.js 16 (App Router) + React 19
- **Langage** : TypeScript
- **Styling** : Tailwind CSS 4
- **Icons** : Lucide React
- **Traitement d'image** : Canvas API (Browser native)

### Architecture de traitement

#### Module A - Débruitage
```
Image d'entrée
    ↓
Filtre bilatéral
    ├─ Préservation des contours (spatial weight)
    └─ Lissage adaptatif (range weight)
    ↓
Amélioration du contraste
    └─ Ajustement adaptatif autour du point médian
    ↓
Calcul des métriques
    ├─ Variance du bruit
    └─ Plage de contraste
    ↓
Image améliorée
```

**Algorithmes utilisés** :
- **Filtre bilatéral** : Réduit le bruit tout en préservant les contours
  - Paramètres : σ_space = 5.0, σ_range = 30.0, kernel radius = 3
- **Enhancement de contraste** : Amélioration multiplicative autour du point médian
  - Facteur : 1.3 (ajustable)

#### Module B - Segmentation (version démonstrative)
```
Image d'entrée
    ↓
Conversion en niveaux de gris
    ↓
Seuillage d'intensité
    └─ Détection des zones haute densité (> 180)
    ↓
Coloration des structures détectées
    ↓
Image segmentée
```

### Structure du code

```
coro-plus-ai/
├── app/
│   ├── layout.tsx          # Layout principal avec métadonnées
│   ├── page.tsx            # Interface utilisateur principale
│   └── globals.css         # Styles globaux
├── lib/
│   └── imageProcessing.ts  # Algorithmes de traitement d'image
├── public/                 # Assets statiques
├── package.json
└── README.md
```

---

## 📊 Métriques et évaluation

### Métriques calculées

1. **Réduction du bruit** : Basée sur la variance des pixels
   - Formule : `(variance_original - variance_processed) / variance_original * 100`

2. **Amélioration du contraste** : Basée sur la plage dynamique
   - Formule : `(contrast_processed - contrast_original) / contrast_original * 100`

3. **Temps de traitement** : Mesure de performance en millisecondes

### Résultats attendus

| Métrique | Valeur typique | Objectif |
|----------|---------------|----------|
| Réduction du bruit | 15-30% | > 20% |
| Amélioration contraste | 20-40% | > 15% |
| Temps de traitement | 100-500ms | < 1000ms |

---

## 🔬 Intégration avec modèles Deep Learning

### Pour passer en production

Le système actuel utilise des algorithmes classiques de traitement d'image pour la démonstration. Pour une application clinique, il est recommandé d'intégrer des modèles de deep learning :

#### Architecture recommandée pour Module A (Débruitage)

```python
# Autoencodeur pour débruitage
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

#### Architecture recommandée pour Module B (Segmentation)

```python
# U-Net pour segmentation coronaires
class CoronaryUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
```

### Datasets recommandés pour entraînement

- **ASOCA** (Automated Segmentation of Coronary Arteries)
- **CA-500** (Coronary Artery 500)
- Données synthétiques générées avec bruit ajouté

### Intégration dans Next.js

Option 1 : **API Backend Python**
```typescript
// Créer une API route Next.js
// app/api/denoise/route.ts
export async function POST(request: Request) {
  const formData = await request.formData();
  const image = formData.get('image');
  
  // Appeler API Python
  const response = await fetch('http://localhost:5000/denoise', {
    method: 'POST',
    body: formData
  });
  
  return response;
}
```

Option 2 : **TensorFlow.js (in-browser)**
```typescript
import * as tf from '@tensorflow/tfjs';

async function loadModel() {
  const model = await tf.loadLayersModel('/models/denoising/model.json');
  return model;
}

async function denoise(imageData: ImageData) {
  const model = await loadModel();
  const tensor = tf.browser.fromPixels(imageData, 1);
  const normalized = tensor.div(255.0);
  const batched = normalized.expandDims(0);
  const prediction = model.predict(batched);
  return prediction;
}
```

---

## ⚠️ Limitations et avertissements

### Limitations actuelles

1. **Usage académique uniquement** : Ce prototype n'est pas validé cliniquement
2. **Pas de support DICOM natif** : Conversion manuelle nécessaire
3. **Algorithmes simplifiés** : Les algorithmes actuels sont démonstratifs
4. **Pas de validation médicale** : Non testé sur des cas cliniques réels
5. **Segmentation basique** : Module B utilise des seuils simples

### Avertissements importants

⚠️ **Ce système ne doit PAS être utilisé pour :**
- Le diagnostic médical clinique
- La prise de décision thérapeutique
- Le remplacement de l'expertise médicale

✅ **Ce système peut être utilisé pour :**
- Démonstration pédagogique
- Recherche académique
- Exploration de concepts IA en imagerie médicale
- Base pour développement ultérieur

---

## 📈 Développements futurs

### Roadmap proposée

#### Phase 1 - MVP actuel ✅
- [x] Interface web fonctionnelle
- [x] Module A : Débruitage basique
- [x] Module B : Segmentation démonstrative
- [x] Génération de rapport

#### Phase 2 - Amélioration IA
- [ ] Intégration modèle PyTorch/TensorFlow
- [ ] Entraînement sur dataset ASOCA
- [ ] Support DICOM natif
- [ ] Optimisation des performances

#### Phase 3 - Fonctionnalités avancées
- [ ] Analyse quantitative des sténoses
- [ ] Calcul de la FFR (Fractional Flow Reserve)
- [ ] Visualisation 3D des coronaires
- [ ] Export DICOM avec métadonnées

#### Phase 4 - Validation clinique
- [ ] Tests sur cas réels
- [ ] Validation par radiologues/cardiologues
- [ ] Étude comparative avec méthodes standard
- [ ] Publication scientifique

---

## 🤝 Contribution et collaboration

### Contributeurs

- **Abidine** - Étudiant L3, Manipulateur en Imagerie Médicale, INFSPM Oran
- Développé avec le support d'outils d'IA

### Comment contribuer

Pour contribuer à ce projet académique :

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Contact

Pour questions ou collaborations :
- Institution : INFSPM Oran
- Projet : Coro-Plus AI
- Type : Projet académique L3

---

## 📚 Références

### Articles scientifiques pertinents

1. **Deep Learning for Image Denoising:**
   - Zhang et al. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"

2. **Medical Image Segmentation:**
   - Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

3. **Coronary CT Analysis:**
   - Lessmann et al. (2019). "Automatic Calcium Scoring in Low-Dose Chest CT Using Deep Neural Networks"

### Ressources techniques

- [Next.js Documentation](https://nextjs.org/docs)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [PyTorch](https://pytorch.org/)
- [DICOM Standard](https://www.dicomstandard.org/)

---

## 📄 License

Ce projet est développé à des fins académiques dans le cadre d'un projet de fin d'études. 

**Utilisation académique uniquement - Pas d'usage clinique**

---

## 🙏 Remerciements

- INFSPM Oran - Institut National de Formation Supérieure Paramédicale
- Encadrants académiques
- Communauté open-source (Next.js, React, TensorFlow)

---

**Version** : 0.1.0 (MVP)  
**Date** : Décembre 2024  
**Statut** : Prototype académique  

---

*Coro-Plus AI - Améliorer l'imagerie coronaire par l'Intelligence Artificielle*
