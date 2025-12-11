# Démarrage Rapide - Coro-Plus AI

## 🚀 Lancer l'application en 3 étapes

### 1. Installation des dépendances

```bash
npm install
```

Cette commande installe toutes les dépendances nécessaires (Next.js, React, Tailwind CSS, etc.)

### 2. Lancer le serveur de développement

```bash
npm run dev
```

L'application sera accessible à : **http://localhost:3000**

### 3. Tester l'application

1. Ouvrir votre navigateur à `http://localhost:3000`
2. Préparer une image de test (PNG ou JPEG)
3. Cliquer dans la zone de dépôt pour charger l'image
4. Sélectionner "Module A - Débruitage"
5. Cliquer sur "Appliquer le débruitage IA"
6. Observer les résultats avant/après
7. Télécharger les images et générer le rapport

---

## 📁 Où trouver des images de test ?

### Option 1 : Banques d'images médicales

- **Radiopaedia** : https://radiopaedia.org/ (rechercher "coronary CT")
- **The Cancer Imaging Archive** : https://www.cancerimagingarchive.net/
- **MedPix** : https://medpix.nlm.nih.gov/

### Option 2 : Convertir des fichiers DICOM

Si vous avez des fichiers DICOM (.dcm), convertissez-les en PNG :

**Avec Python :**
```python
import pydicom
from PIL import Image

ds = pydicom.dcmread('scan.dcm')
pixels = ds.pixel_array
Image.fromarray(pixels).save('scan.png')
```

**Avec RadiAnt Viewer :**
1. Ouvrir le fichier DICOM
2. Fichier → Exporter → Image PNG
3. Sauvegarder

### Option 3 : Images simulées

Pour tester rapidement, vous pouvez utiliser n'importe quelle image médicale (radiographie, scanner) ou même des images générales pour voir le débruitage en action.

---

## 🎯 Cas d'usage rapide

### Test 1 : Débruitage (2 minutes)

1. Charger une image légèrement bruitée
2. Module A → "Appliquer le débruitage IA"
3. Observer :
   - Réduction du grain
   - Amélioration du contraste
   - Métriques (temps, réduction bruit %, contraste %)
4. Télécharger l'image améliorée

### Test 2 : Segmentation (1 minute)

1. Charger une image de coroscanner
2. Module B → "Détecter les structures vasculaires"
3. Observer :
   - Structures haute densité en rouge
   - Zone vasculaire mise en évidence
4. Télécharger l'image segmentée

### Test 3 : Rapport complet (3 minutes)

1. Effectuer un débruitage (Test 1)
2. Cliquer sur "Générer le rapport complet (.txt)"
3. Ouvrir le fichier téléchargé
4. Observer :
   - Métriques détaillées
   - Interprétation automatique
   - Recommandations

---

## 🛠️ Commandes utiles

### Développement
```bash
npm run dev          # Lancer en mode développement
```

### Production
```bash
npm run build        # Créer une version optimisée
npm start            # Lancer la version de production
```

### Qualité du code
```bash
npm run lint         # Vérifier le code (ESLint)
```

### Autres
```bash
npm install          # Installer/Réinstaller les dépendances
node --version       # Vérifier la version de Node.js (doit être 20+)
```

---

## 📊 Résultats attendus

### Module A - Débruitage

| Métrique | Valeur attendue | Interprétation |
|----------|-----------------|----------------|
| Temps de traitement | 200-600 ms | Traitement rapide |
| Réduction du bruit | 15-30% | Amélioration significative |
| Amélioration contraste | 20-40% | Meilleure lisibilité |

### Module B - Segmentation

- Structures vasculaires surlignées en rouge
- Zones haute densité détectées
- Visualisation immédiate

---

## ⚠️ Problèmes courants

### Erreur : "Port 3000 already in use"

**Solution :**
```bash
# Option 1 : Tuer le processus
kill -9 $(lsof -ti:3000)

# Option 2 : Utiliser un autre port
PORT=3001 npm run dev
```

### Erreur : "Module not found"

**Solution :**
```bash
rm -rf node_modules package-lock.json
npm install
```

### L'image ne se charge pas

**Vérifications :**
- Format : PNG ou JPEG uniquement
- Taille : < 10 MB recommandé
- Navigateur compatible (Chrome, Firefox, Edge)

### Le traitement est très lent

**Causes possibles :**
- Image trop grande (> 2000×2000 px)
- Ordinateur lent
- Navigateur ancien

**Solution :** Redimensionner l'image à 512×512 ou 1024×1024 pixels

---

## 📱 Compatibilité

### Navigateurs supportés
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

### Systèmes d'exploitation
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu, Debian, Fedora)

### Node.js
- ✅ Version 20 ou supérieure recommandée
- ⚠️ Version 18 minimum requise

---

## 📚 Documentation complète

Pour plus d'informations :

- **README.md** : Documentation technique complète
- **GUIDE_UTILISATEUR.md** : Guide d'utilisation détaillé
- **GUIDE_INTEGRATION_IA.md** : Guide pour intégrer le Deep Learning
- **PROJET_RESUME.md** : Résumé du projet pour mémoire

---

## 🎓 Pour la démonstration

### Préparer en 10 minutes

1. **Installer** (2 min)
   ```bash
   npm install
   ```

2. **Tester** (3 min)
   ```bash
   npm run dev
   ```
   Ouvrir http://localhost:3000 et tester avec 2-3 images

3. **Prendre des captures d'écran** (5 min)
   - Page d'accueil
   - Module A avant/après
   - Module B avec segmentation
   - Rapport généré

### Scénario de démonstration (5 minutes)

**Minute 1 :** Introduction
> "Coro-Plus AI est un système d'IA pour améliorer les images de coroscanner..."

**Minute 2 :** Chargement et Module A
> "Je charge une image... J'applique le débruitage... Voici les résultats..."

**Minute 3 :** Métriques et analyse
> "On observe une réduction du bruit de 25%, amélioration du contraste de 30%..."

**Minute 4 :** Module B
> "La segmentation détecte automatiquement les structures vasculaires..."

**Minute 5 :** Rapport et conclusion
> "Le système génère un rapport complet... Perspectives d'évolution..."

---

## ✅ Checklist de démarrage

Avant la démonstration ou présentation :

- [ ] Node.js 20+ installé (`node --version`)
- [ ] Dépendances installées (`npm install`)
- [ ] Application démarre sans erreur (`npm run dev`)
- [ ] 2-3 images de test préparées
- [ ] Test réussi de Module A (débruitage)
- [ ] Test réussi de Module B (segmentation)
- [ ] Rapport généré au moins une fois
- [ ] Captures d'écran prises
- [ ] Documentation lue (README.md minimum)

---

## 🚀 Et après ?

Une fois l'application lancée et testée :

1. **Explorer** : Tester avec différents types d'images
2. **Analyser** : Comparer les métriques sur plusieurs cas
3. **Documenter** : Intégrer les résultats dans votre mémoire
4. **Améliorer** : Consulter GUIDE_INTEGRATION_IA.md pour évolutions

---

## 💬 Support

En cas de problème :

1. **Lire** : GUIDE_UTILISATEUR.md (section "En cas de problème")
2. **Vérifier** : Node.js version, dépendances installées
3. **Tester** : Réinstaller les dépendances
4. **Rechercher** : Erreur sur Google/Stack Overflow

---

**Bon démarrage avec Coro-Plus AI ! 🎉**

*Version 0.1.0 - MVP Académique*
