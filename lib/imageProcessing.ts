export interface ProcessingResult {
  originalImage: string;
  denoisedImage: string;
  processingTime: number;
  noiseReduction: number;
  contrastImprovement: number;
}

export async function applyDenoising(imageFile: File): Promise<ProcessingResult> {
  const startTime = performance.now();
  
  const originalDataUrl = await fileToDataUrl(imageFile);
  const img = await loadImage(originalDataUrl);
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  
  if (!ctx) {
    throw new Error('Canvas context not available');
  }
  
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
  const denoisedData = bilateralFilter(imageData);
  
  const enhancedData = enhanceContrast(denoisedData);
  
  ctx.putImageData(enhancedData, 0, 0);
  
  const denoisedImage = canvas.toDataURL('image/png');
  const processingTime = performance.now() - startTime;
  
  const noiseReduction = calculateNoiseReduction(imageData, enhancedData);
  const contrastImprovement = calculateContrastImprovement(imageData, enhancedData);
  
  return {
    originalImage: originalDataUrl,
    denoisedImage,
    processingTime,
    noiseReduction,
    contrastImprovement,
  };
}

function bilateralFilter(imageData: ImageData): ImageData {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const output = new ImageData(width, height);
  const outputData = output.data;
  
  const kernelRadius = 3;
  const sigmaSpace = 5.0;
  const sigmaRange = 30.0;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      
      let sumR = 0, sumG = 0, sumB = 0;
      let totalWeight = 0;
      
      const centerR = data[idx];
      const centerG = data[idx + 1];
      const centerB = data[idx + 2];
      
      for (let ky = -kernelRadius; ky <= kernelRadius; ky++) {
        for (let kx = -kernelRadius; kx <= kernelRadius; kx++) {
          const nx = Math.max(0, Math.min(width - 1, x + kx));
          const ny = Math.max(0, Math.min(height - 1, y + ky));
          const nIdx = (ny * width + nx) * 4;
          
          const spatialDist = kx * kx + ky * ky;
          const spatialWeight = Math.exp(-spatialDist / (2 * sigmaSpace * sigmaSpace));
          
          const diffR = data[nIdx] - centerR;
          const diffG = data[nIdx + 1] - centerG;
          const diffB = data[nIdx + 2] - centerB;
          const rangeDist = diffR * diffR + diffG * diffG + diffB * diffB;
          const rangeWeight = Math.exp(-rangeDist / (2 * sigmaRange * sigmaRange));
          
          const weight = spatialWeight * rangeWeight;
          
          sumR += data[nIdx] * weight;
          sumG += data[nIdx + 1] * weight;
          sumB += data[nIdx + 2] * weight;
          totalWeight += weight;
        }
      }
      
      outputData[idx] = Math.round(sumR / totalWeight);
      outputData[idx + 1] = Math.round(sumG / totalWeight);
      outputData[idx + 2] = Math.round(sumB / totalWeight);
      outputData[idx + 3] = data[idx + 3];
    }
  }
  
  return output;
}

function enhanceContrast(imageData: ImageData, factor: number = 1.3): ImageData {
  const data = imageData.data;
  const output = new ImageData(imageData.width, imageData.height);
  const outputData = output.data;
  
  for (let i = 0; i < data.length; i += 4) {
    outputData[i] = clamp((data[i] - 128) * factor + 128);
    outputData[i + 1] = clamp((data[i + 1] - 128) * factor + 128);
    outputData[i + 2] = clamp((data[i + 2] - 128) * factor + 128);
    outputData[i + 3] = data[i + 3];
  }
  
  return output;
}

function calculateNoiseReduction(original: ImageData, processed: ImageData): number {
  const originalVariance = calculateVariance(original);
  const processedVariance = calculateVariance(processed);
  
  const reduction = ((originalVariance - processedVariance) / originalVariance) * 100;
  return Math.max(0, Math.min(100, reduction));
}

function calculateContrastImprovement(original: ImageData, processed: ImageData): number {
  const originalContrast = calculateContrast(original);
  const processedContrast = calculateContrast(processed);
  
  const improvement = ((processedContrast - originalContrast) / originalContrast) * 100;
  return Math.max(0, Math.min(100, improvement));
}

function calculateVariance(imageData: ImageData): number {
  const data = imageData.data;
  let sum = 0;
  let count = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
    sum += gray;
    count++;
  }
  
  const mean = sum / count;
  let variance = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
    variance += Math.pow(gray - mean, 2);
  }
  
  return variance / count;
}

function calculateContrast(imageData: ImageData): number {
  const data = imageData.data;
  let min = 255;
  let max = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
    min = Math.min(min, gray);
    max = Math.max(max, gray);
  }
  
  return max - min;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

export async function simulateSegmentation(imageFile: File): Promise<string> {
  const dataUrl = await fileToDataUrl(imageFile);
  const img = await loadImage(dataUrl);
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  if (!ctx) {
    throw new Error('Canvas context not available');
  }
  
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
    
    if (gray > 180) {
      data[i] = Math.min(255, data[i] + 30);
      data[i + 1] = Math.min(255, data[i + 1] - 30);
      data[i + 2] = Math.min(255, data[i + 2] - 30);
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  
  return canvas.toDataURL('image/png');
}
