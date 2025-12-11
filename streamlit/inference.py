import time
import numpy as np
from PIL import Image, ImageFilter

def add_noise(image: Image.Image, strength: float) -> Image.Image:
    """
    Add Gaussian noise to the image.
    strength: float between 0 and 1, where 1 is maximum noise.
    """
    if strength == 0:
        return image.copy()
        
    img_array = np.array(image).astype(float)
    # Scale strength to a reasonable standard deviation for pixel values
    sigma = strength * 100 
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

class DenoiseModel:
    def __init__(self):
        # Simulate model loading time
        time.sleep(2.0)
        self.loaded = True

    def denoise(self, noisy_image: Image.Image) -> Image.Image:
        """
        Simulate denoising.
        In a real scenario, this would use a deep learning model.
        Here, we use a simple Median Filter to demonstrate the concept of removing noise.
        """
        # Simulate inference time
        time.sleep(1.0)
        
        # Apply a median filter which is good for removing salt-and-pepper noise,
        # and decent for gaussian noise visualization in a demo.
        # Using a fixed size for simplicity.
        return noisy_image.filter(ImageFilter.MedianFilter(size=5))
