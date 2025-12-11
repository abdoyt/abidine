import os
import time
import streamlit as st
import numpy as np
from PIL import Image
from inference import DenoiseModel, add_noise

# Set page config
st.set_page_config(
    page_title="Medical Image Denoising Demo",
    page_icon="🏥",
    layout="wide"
)

# Constants
EXAMPLES_DIR = "examples"
os.makedirs(EXAMPLES_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    """Load the denoising model (cached)."""
    with st.spinner("Loading Denoising Model..."):
        return DenoiseModel()

def get_sample_image():
    """Generate a sample synthetic image (Shepp-Logan-like or simple geometric)."""
    # Create a 256x256 grayscale image
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Draw some shapes
    # Circle
    y, x = np.ogrid[:size, :size]
    mask_circle = (x - size//2)**2 + (y - size//2)**2 <= (size//3)**2
    img[mask_circle] = 200
    
    # Rectangle inside
    img[size//2-30:size//2+30, size//2-60:size//2+60] = 100
    
    # Another circle
    mask_small = (x - size//2 - 40)**2 + (y - size//2 - 40)**2 <= (size//8)**2
    img[mask_small] = 255
    
    return Image.fromarray(img)

def save_image(image: Image.Image, filename: str):
    path = os.path.join(EXAMPLES_DIR, filename)
    image.save(path)
    return path

def main():
    st.title("Medical Image Denoising & Segmentation Demo")
    st.markdown("""
    This app demonstrates low-dose simulation (noise addition) and AI-based denoising.
    Future versions will include Module B segmentation overlays.
    """)

    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # 1. Input Source
    input_source = st.sidebar.radio("Select Input Source", ["Upload Image", "Use Bundled Sample"])
    
    original_image = None
    
    if input_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tif"])
        if uploaded_file is not None:
            try:
                original_image = Image.open(uploaded_file).convert("L") # Convert to grayscale
            except Exception as e:
                st.sidebar.error(f"Error loading image: {e}")
    else:
        # Default to sample if selected
        original_image = get_sample_image()
        st.sidebar.success("Using Bundled Sample")

    # 2. Noise Simulation
    noise_strength = st.sidebar.slider("Noise Simulation Strength (Low Dose)", 0.0, 1.0, 0.2, 0.05)
    
    # 3. Toggle for Segmentation (Placeholder)
    show_segmentation = st.sidebar.toggle("Show Module B Segmentation Overlays (Coming Soon)")

    if show_segmentation:
        st.sidebar.info("Segmentation module is currently under development.")

    # Main content
    if original_image:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Input")
            st.image(original_image, use_container_width=True, caption="Ground Truth")

        # Apply Noise
        noisy_image = add_noise(original_image, noise_strength)
        
        with col2:
            st.subheader("Simulated Low-Dose")
            st.image(noisy_image, use_container_width=True, caption=f"Noisy Input (Strength: {noise_strength})")

        # Denoise
        model = load_model()
        
        # Button to run inference (or run automatically)
        # Ticket says "UI can run on CPU within seconds". Auto-run is better for interactivity.
        
        start_time = time.time()
        denoised_image = model.denoise(noisy_image)
        end_time = time.time()
        
        with col3:
            st.subheader("AI Denoised Output")
            st.image(denoised_image, use_container_width=True, caption=f"Result ({end_time - start_time:.2f}s)")
            
        # Download results
        st.divider()
        st.subheader("Downloads")
        
        # Save to temporary examples dir for report (as per ticket)
        timestamp = int(time.time())
        noisy_path = save_image(noisy_image, f"noisy_{timestamp}.png")
        denoised_path = save_image(denoised_image, f"denoised_{timestamp}.png")
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            with open(noisy_path, "rb") as file:
                st.download_button(
                    label="Download Noisy Image",
                    data=file,
                    file_name="noisy_input.png",
                    mime="image/png"
                )
        with d_col2:
            with open(denoised_path, "rb") as file:
                st.download_button(
                    label="Download Denoised Image",
                    data=file,
                    file_name="denoised_output.png",
                    mime="image/png"
                )
                
        if show_segmentation:
            st.warning("Overlay visualization not yet implemented.")

    else:
        st.info("Please upload an image or select the bundled sample to start.")

if __name__ == "__main__":
    main()
