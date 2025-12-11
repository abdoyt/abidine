'use client';

import { useState } from 'react';
import { Upload, Image as ImageIcon, Download, Activity, Zap, FileText } from 'lucide-react';
import { applyDenoising, simulateSegmentation, type ProcessingResult } from '@/lib/imageProcessing';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'denoising' | 'segmentation'>('denoising');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setResult(null);
      setSegmentedImage(null);
      setShowSegmentation(false);
    }
  };

  const handleProcess = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    try {
      if (activeTab === 'denoising') {
        const processingResult = await applyDenoising(selectedFile);
        setResult(processingResult);
      } else {
        const segmented = await simulateSegmentation(selectedFile);
        setSegmentedImage(segmented);
        setShowSegmentation(true);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Erreur lors du traitement de l\'image');
    } finally {
      setProcessing(false);
    }
  };

  const downloadImage = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    link.click();
  };

  const generateReport = () => {
    if (!result) return;

    const report = `
RAPPORT D'ANALYSE - CORO-PLUS AI
================================

Date: ${new Date().toLocaleDateString('fr-FR')}
Heure: ${new Date().toLocaleTimeString('fr-FR')}

MODULE A: AMÉLIORATION D'IMAGE (DÉBRUITAGE)
-------------------------------------------

Résultats du traitement:
- Temps de traitement: ${result.processingTime.toFixed(2)} ms
- Réduction du bruit: ${result.noiseReduction.toFixed(1)}%
- Amélioration du contraste: ${result.contrastImprovement.toFixed(1)}%

Méthode utilisée:
- Filtre bilatéral pour préservation des contours
- Amélioration adaptative du contraste
- Normalisation de l'intensité

Interprétation:
${result.noiseReduction > 20 ? '✓ Bonne réduction du bruit obtenue' : '○ Réduction du bruit modérée'}
${result.contrastImprovement > 15 ? '✓ Amélioration significative du contraste' : '○ Amélioration modérée du contraste'}

Recommandations:
- Images améliorées adaptées pour analyse détaillée
- Qualité diagnostique préservée
- Réduction théorique de la dose possible

================================
Généré par Coro-Plus AI v0.1.0
Projet académique - INFSPM Oran
================================
    `.trim();

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `rapport-coroplus-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-950 dark:to-blue-950">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-12 h-12 text-blue-600 dark:text-blue-400" />
            <h1 className="text-5xl font-bold text-slate-900 dark:text-slate-50">
              Coro-Plus AI
            </h1>
          </div>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
            Système IA pour l&apos;amélioration du coroscanner en imagerie coronaire
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-500 mt-2">
            Projet académique - INFSPM Oran - Abidine, L3 Manipulateur en Imagerie Médicale
          </p>
        </header>

        <div className="max-w-6xl mx-auto">
          <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-xl p-8 mb-8">
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="bg-blue-50 dark:bg-blue-950/30 rounded-xl p-6 border-2 border-blue-200 dark:border-blue-800">
                <Zap className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-3" />
                <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-2">
                  Module A: Débruitage
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Amélioration d&apos;image pour réduction du bruit et optimisation du contraste
                </p>
              </div>

              <div className="bg-green-50 dark:bg-green-950/30 rounded-xl p-6 border-2 border-green-200 dark:border-green-800">
                <ImageIcon className="w-8 h-8 text-green-600 dark:text-green-400 mb-3" />
                <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-2">
                  Module B: Segmentation
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Détection basique des structures vasculaires coronaires
                </p>
              </div>

              <div className="bg-purple-50 dark:bg-purple-950/30 rounded-xl p-6 border-2 border-purple-200 dark:border-purple-800">
                <FileText className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-3" />
                <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-2">
                  Rapport automatique
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Génération de rapport d&apos;analyse avec métriques quantitatives
                </p>
              </div>
            </div>

            <div className="border-b border-slate-200 dark:border-slate-700 mb-6">
              <div className="flex gap-4">
                <button
                  onClick={() => setActiveTab('denoising')}
                  className={`px-6 py-3 font-medium border-b-2 transition-colors ${
                    activeTab === 'denoising'
                      ? 'border-blue-600 text-blue-600 dark:border-blue-400 dark:text-blue-400'
                      : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  Module A - Débruitage
                </button>
                <button
                  onClick={() => setActiveTab('segmentation')}
                  className={`px-6 py-3 font-medium border-b-2 transition-colors ${
                    activeTab === 'segmentation'
                      ? 'border-green-600 text-green-600 dark:border-green-400 dark:text-green-400'
                      : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  Module B - Segmentation
                </button>
              </div>
            </div>

            <div className="mb-8">
              <label className="block mb-4">
                <div className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-12 text-center hover:border-blue-400 dark:hover:border-blue-600 transition-colors cursor-pointer bg-slate-50 dark:bg-slate-800/50">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <Upload className="w-12 h-12 text-slate-400 dark:text-slate-600 mx-auto mb-4" />
                  <p className="text-lg font-medium text-slate-700 dark:text-slate-300 mb-2">
                    {selectedFile ? selectedFile.name : 'Sélectionner une image de coroscanner'}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-500">
                    PNG, JPEG (Format DICOM à convertir en PNG/JPEG)
                  </p>
                </div>
              </label>

              {selectedFile && (
                <button
                  onClick={handleProcess}
                  disabled={processing}
                  className={`w-full py-4 rounded-xl font-semibold text-white text-lg transition-all ${
                    processing
                      ? 'bg-slate-400 dark:bg-slate-600 cursor-not-allowed'
                      : activeTab === 'denoising'
                      ? 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'
                      : 'bg-green-600 hover:bg-green-700 dark:bg-green-500 dark:hover:bg-green-600'
                  }`}
                >
                  {processing ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
                      Traitement en cours...
                    </span>
                  ) : activeTab === 'denoising' ? (
                    'Appliquer le débruitage IA'
                  ) : (
                    'Détecter les structures vasculaires'
                  )}
                </button>
              )}
            </div>

            {result && activeTab === 'denoising' && (
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <h3 className="font-semibold text-slate-900 dark:text-slate-50 flex items-center gap-2">
                      <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                      Image originale
                    </h3>
                    <div className="relative aspect-square bg-slate-100 dark:bg-slate-800 rounded-xl overflow-hidden border-2 border-slate-200 dark:border-slate-700">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={result.originalImage}
                        alt="Original"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <button
                      onClick={() => downloadImage(result.originalImage, 'original.png')}
                      className="w-full py-2 px-4 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg flex items-center justify-center gap-2 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Télécharger
                    </button>
                  </div>

                  <div className="space-y-3">
                    <h3 className="font-semibold text-slate-900 dark:text-slate-50 flex items-center gap-2">
                      <span className="w-3 h-3 bg-green-500 rounded-full"></span>
                      Image améliorée (IA)
                    </h3>
                    <div className="relative aspect-square bg-slate-100 dark:bg-slate-800 rounded-xl overflow-hidden border-2 border-green-500 dark:border-green-600">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={result.denoisedImage}
                        alt="Denoised"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <button
                      onClick={() => downloadImage(result.denoisedImage, 'denoised.png')}
                      className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center justify-center gap-2 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Télécharger
                    </button>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    Résultats du traitement
                  </h3>
                  <div className="grid sm:grid-cols-3 gap-4">
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Temps de traitement</p>
                      <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {result.processingTime.toFixed(0)} ms
                      </p>
                    </div>
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Réduction du bruit</p>
                      <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {result.noiseReduction.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Amélioration contraste</p>
                      <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                        {result.contrastImprovement.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                <button
                  onClick={generateReport}
                  className="w-full py-3 px-6 bg-slate-900 hover:bg-slate-800 dark:bg-slate-700 dark:hover:bg-slate-600 text-white rounded-xl flex items-center justify-center gap-2 transition-colors font-medium"
                >
                  <FileText className="w-5 h-5" />
                  Générer le rapport complet (.txt)
                </button>
              </div>
            )}

            {showSegmentation && segmentedImage && activeTab === 'segmentation' && (
              <div className="space-y-6">
                <div className="space-y-3">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-50 flex items-center gap-2">
                    <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                    Segmentation des structures vasculaires
                  </h3>
                  <div className="relative aspect-video bg-slate-100 dark:bg-slate-800 rounded-xl overflow-hidden border-2 border-green-500 dark:border-green-600">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={segmentedImage}
                      alt="Segmented"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <button
                    onClick={() => downloadImage(segmentedImage, 'segmented.png')}
                    className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center justify-center gap-2 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Télécharger l&apos;image segmentée
                  </button>
                </div>

                <div className="bg-green-50 dark:bg-green-950/30 rounded-xl p-6 border border-green-200 dark:border-green-800">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-3">
                    ⚠️ Note sur la segmentation
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Cette version démontre le concept de segmentation. Pour une segmentation médicale précise,
                    il faudrait intégrer un modèle U-Net entraîné sur des données coronaires réelles.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-50 mb-6">
              À propos du projet
            </h2>
            <div className="prose prose-slate dark:prose-invert max-w-none">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-50">Objectifs</h3>
              <p className="text-slate-600 dark:text-slate-400">
                Coro-Plus AI est un prototype académique développé pour démontrer le potentiel de l&apos;Intelligence
                Artificielle dans l&apos;amélioration des images de coroscanner. Le système vise à réduire le bruit,
                améliorer le contraste, et faciliter la détection des structures coronaires.
              </p>

              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-50 mt-6">Architecture technique</h3>
              <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 mt-2">
                <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-2">
                  <li><strong>Module A (Débruitage):</strong> Filtre bilatéral avec préservation des contours + amélioration adaptative du contraste</li>
                  <li><strong>Module B (Segmentation):</strong> Détection basée sur les seuils d&apos;intensité (version démonstrative)</li>
                  <li><strong>Pour production:</strong> Intégration recommandée avec modèles PyTorch/TensorFlow (U-Net, Autoencodeur)</li>
                </ul>
              </div>

              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-50 mt-6">Limitations</h3>
              <p className="text-slate-600 dark:text-slate-400">
                Ce prototype est destiné à des fins pédagogiques et de recherche uniquement. Il ne doit pas être
                utilisé pour le diagnostic médical clinique. Les algorithmes actuels sont des approximations
                pour démonstration et nécessiteraient validation clinique pour usage réel.
              </p>

              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-50 mt-6">Développement futur</h3>
              <ul className="text-slate-600 dark:text-slate-400 list-disc list-inside space-y-1">
                <li>Intégration de modèles deep learning pré-entraînés</li>
                <li>Support natif du format DICOM</li>
                <li>Analyse quantitative des sténoses</li>
                <li>Estimation de la FFR (Fractional Flow Reserve)</li>
                <li>Visualisation 3D des artères coronaires</li>
              </ul>
            </div>
          </div>
        </div>

        <footer className="text-center mt-12 text-slate-500 dark:text-slate-500 text-sm">
          <p>© 2024 Coro-Plus AI - Projet académique L3 Manipulateur en Imagerie Médicale</p>
          <p className="mt-1">INFSPM Oran - Version MVP 0.1.0</p>
        </footer>
      </div>
    </div>
  );
}
