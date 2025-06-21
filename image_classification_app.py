import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# CIFAR-10 class mapping (10 classes)
CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile", 
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Load custom trained MobileNetV1 model
@st.cache_resource
def load_custom_mobilenet():
    """Load custom trained MobileNetV1 model for CIFAR-10"""
    model_path = os.path.join("models", "mobilenet_cifar10_best.keras")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_cifar10_predictions(predictions):
    """Get predictions for CIFAR-10 classes"""
    cifar10_preds = []
    
    for idx, confidence in enumerate(predictions):
        if idx < len(CIFAR10_CLASSES):
            cifar10_preds.append({
                'index': idx,
                'class': CIFAR10_CLASSES[idx],
                'confidence': confidence
            })
    
    # Sort by confidence
    cifar10_preds.sort(key=lambda x: x['confidence'], reverse=True)
    return cifar10_preds

# Page config
st.set_page_config(
    page_title="Custom MobileNetV1 CIFAR-10 Classification",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 Custom MobileNetV1 CIFAR-10 Classification")
    st.markdown("*Upload an image and get class predictions from your custom trained MobileNetV1 model (10 CIFAR-10 classes)*")
    
    # Load model
    with st.spinner("Loading your custom MobileNetV1 model..."):
        model = load_custom_mobilenet()
    
    if model is None:
        st.error("❌ Failed to load the custom model. Please check if 'models/mobilenet_cifar10_best.keras' exists.")
        return
    
    # Sidebar for model info
    st.sidebar.title("⚙️ Model Information")
    st.sidebar.success("✅ Custom model loaded successfully!")
    st.sidebar.write(f"**Input Size:** 32 × 32 pixels (CIFAR-10 format)")
    st.sidebar.write(f"**Classes:** 10 (CIFAR-10)")
    st.sidebar.write(f"**Parameters:** {model.count_params():,}")
    
    # Show CIFAR-10 classes
    st.sidebar.subheader("📋 CIFAR-10 Classes")
    for i, (idx, class_name) in enumerate(CIFAR10_CLASSES.items(), 1):
        st.sidebar.write(f"{i}. {class_name}")
    
    # Prediction settings
    st.sidebar.subheader("🎯 Prediction Settings")
    num_predictions = st.sidebar.slider(
        "Number of Top Predictions", 1, 10, 5, 1,
        help="Show top N predictions (max 10 for CIFAR-10)"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify (works best with: vehicles, animals, aircraft, etc.)"
        )
        
        if uploaded_image:
            # Display original image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to numpy array and handle different formats
            img_array = np.array(image)
            
            # Handle different image formats
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 3:
                    pass
                else:
                    st.error(f"Unsupported image format: {img_array.shape[2]} channels")
                    return
            else:
                st.error("Invalid image format")
                return
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.write(f"**Original Size:** {img_array.shape[1]} × {img_array.shape[0]} pixels")
            st.write(f"**Channels:** {img_array.shape[2] if len(img_array.shape) == 3 else 1}")
            st.write(f"**Will be resized to:** 32 × 32 pixels (CIFAR-10 format)")
            
    with col2:
        st.subheader("🎯 Classification Results")
        
        if uploaded_image:
            with st.spinner("Analyzing image..."):
                
                # Resize image to 32x32 (CIFAR-10 input size)
                resized_img = cv2.resize(img_array, (32, 32))
                
                # Preprocess for custom MobileNet
                img_processed = np.expand_dims(resized_img, axis=0)
                
                # Predict
                pred = model.predict(img_processed, verbose=0)
                
                # Get CIFAR-10 predictions
                cifar10_preds = get_cifar10_predictions(pred[0])
                
                st.subheader("📊 Classification Results")
                st.write(f"*Image resized to 32×32 pixels for classification*")
                
                # Show top N predictions
                for i, pred_info in enumerate(cifar10_preds[:num_predictions], 1):
                    # Create a more visual representation
                    confidence_pct = pred_info['confidence'] * 100
                    
                    st.write(f"**{i}. {pred_info['class']}**")
                    st.progress(float(pred_info['confidence']))
                    st.write(f"Confidence: {pred_info['confidence']:.4f} ({confidence_pct:.2f}%)")
                    st.write("---")
                
                # Show the top prediction prominently
                if cifar10_preds:
                    top_pred = cifar10_preds[0]
                    st.subheader("🏆 Top Prediction")
                    st.success(f"**{top_pred['class']}** with {top_pred['confidence']*100:.2f}% confidence")
        
        else:
            st.info("👆 Please upload an image to classify")
    
    # Instructions
    st.subheader("📝 How to Use")
    with st.expander("Click to expand instructions"):
        st.markdown("""
        **CIFAR-10 Classes (10 total):**
        1. **airplane** ✈️
        2. **automobile** 🚗  
        3. **bird** 🐦
        4. **cat** 🐱
        5. **deer** 🦌
        6. **dog** 🐕
        7. **frog** 🐸
        8. **horse** 🐎
        9. **ship** 🚢
        10. **truck** 🚛
        
        **How it works:**
        - Upload any image using the file uploader
        - The image will be automatically resized to 32×32 pixels
        - Your custom MobileNetV1 model will classify it into one of the 10 CIFAR-10 classes
        - You'll see the top predictions with confidence scores
        
        **Best Results:** Upload clear images containing these specific objects for optimal classification!
        
        **Note:** This model was trained on 32×32 pixel images, so it works best with simple objects and scenes.
        """)

if __name__ == "__main__":
    main()
