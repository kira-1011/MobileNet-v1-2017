import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Imagenette class mapping (10 classes)
IMAGENETTE_CLASSES = {
    0: "tench",
    217: "English springer",  # ImageNet index 217
    482: "cassette player",   # ImageNet index 482
    491: "chain saw",         # ImageNet index 491
    497: "church",            # ImageNet index 497
    566: "French horn",       # ImageNet index 566
    569: "garbage truck",     # ImageNet index 569
    571: "gas pump",          # ImageNet index 571
    574: "golf ball",         # ImageNet index 574
    701: "parachute"          # ImageNet index 701
}

# Reverse mapping for easy lookup
IMAGENETTE_INDICES = list(IMAGENETTE_CLASSES.keys())

# Load pre-trained MobileNet from TensorFlow
@st.cache_resource
def load_pretrained_mobilenet():
    """Load pre-trained MobileNetV1 from TensorFlow"""
    from tensorflow.keras.applications import MobileNet
    from tensorflow.keras.applications.mobilenet import preprocess_input
    
    model = MobileNet(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )
    return model, preprocess_input

def filter_imagenette_predictions(predictions):
    """Filter predictions to only show Imagenette classes"""
    # Get predictions for Imagenette indices only
    imagenette_preds = []
    
    for idx in IMAGENETTE_INDICES:
        if idx < len(predictions):
            imagenette_preds.append({
                'index': idx,
                'class': IMAGENETTE_CLASSES[idx],
                'confidence': predictions[idx]
            })
    
    # Sort by confidence
    imagenette_preds.sort(key=lambda x: x['confidence'], reverse=True)
    return imagenette_preds

# Page config
st.set_page_config(
    page_title="MobileNetV1 Imagenette Classification",
    page_icon="🖼️",
    layout="wide"
)

def main():
    st.title("🖼️ MobileNetV1 Imagenette Classification")
    st.markdown("*Upload an image and get class predictions from your MobileNetV1 model (10 Imagenette classes)*")
    
    # Load model and preprocessing functions
    with st.spinner("Loading your MobileNetV1 model..."):
        model, preprocess_input = load_pretrained_mobilenet()
    
    # Sidebar for settings
    st.sidebar.title("⚙️ Settings")
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.write(f"**Input Size:** (224, 224)")
    st.sidebar.write(f"**Classes:** 10 (Imagenette)")
    st.sidebar.write(f"**Parameters:** {model.count_params():,}")
    
    # Show Imagenette classes
    st.sidebar.subheader("📋 Imagenette Classes")
    for i, (idx, class_name) in enumerate(IMAGENETTE_CLASSES.items(), 1):
        st.sidebar.write(f"{i}. {class_name}")
    
    # Prediction settings
    st.sidebar.subheader("🎯 Prediction Settings")
    
    prediction_mode = st.sidebar.selectbox(
        "Prediction Mode:",
        ["whole_image", "center_patch", "multiple_patches"],
        help="How to analyze the image"
    )
    
    num_predictions = st.sidebar.slider(
        "Number of Top Predictions", 1, 10, 5, 1,
        help="Show top N predictions (max 10 for Imagenette)"
    )
    
    st.sidebar.info(f"🔧 Using input shape: 224 × 224")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify (works best with: fish, dogs, electronics, vehicles, etc.)"
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
            st.write(f"**Will be resized to:** 224 × 224 pixels")
            
    with col2:
        st.subheader("🎯 Classification Results")
        
        if uploaded_image:
            with st.spinner("Analyzing image..."):
                
                h, w = img_bgr.shape[:2]
                target_h, target_w = 224, 224
                
                if prediction_mode == "whole_image":
                    # Resize entire image to 224x224
                    resized_img = cv2.resize(img_bgr, (target_w, target_h))
                    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess for MobileNet
                    img_processed = np.expand_dims(resized_img_rgb, axis=0)
                    img_processed = preprocess_input(img_processed)
                    
                    # Predict
                    pred = model.predict(img_processed, verbose=0)
                    
                    # Filter to Imagenette classes
                    imagenette_preds = filter_imagenette_predictions(pred[0])
                    
                    st.subheader("📊 Whole Image Classification")
                    st.write(f"*Resized from {w}×{h} to {target_w}×{target_h}*")
                    
                    # Show top N predictions
                    for i, pred_info in enumerate(imagenette_preds[:num_predictions], 1):
                        st.write(f"**{i}. {pred_info['class']}**")
                        st.progress(float(pred_info['confidence']))
                        st.write(f"Confidence: {pred_info['confidence']:.3f} ({pred_info['confidence']*100:.1f}%)")
                        st.write("---")
                
                elif prediction_mode == "center_patch":
                    # Get center patch
                    half_w, half_h = target_w // 2, target_h // 2
                    center_x, center_y = w//2, h//2
                    
                    start_x = max(0, center_x - half_w)
                    end_x = min(w, center_x + half_w)
                    start_y = max(0, center_y - half_h)
                    end_y = min(h, center_y + half_h)
                    
                    patch = img_bgr[start_y:end_y, start_x:end_x]
                    patch_resized = cv2.resize(patch, (target_w, target_h))
                    patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
                    
                    patch_processed = np.expand_dims(patch_rgb, axis=0)
                    patch_processed = preprocess_input(patch_processed)
                    
                    pred = model.predict(patch_processed, verbose=0)
                    imagenette_preds = filter_imagenette_predictions(pred[0])
                    
                    st.subheader("📊 Center Patch Classification")
                    st.write(f"*Center patch resized to {target_w}×{target_h}*")
                    
                    for i, pred_info in enumerate(imagenette_preds[:num_predictions], 1):
                        st.write(f"**{i}. {pred_info['class']}**")
                        st.progress(float(pred_info['confidence']))
                        st.write(f"Confidence: {pred_info['confidence']:.3f} ({pred_info['confidence']*100:.1f}%)")
                        st.write("---")
                
                elif prediction_mode == "multiple_patches":
                    st.subheader("📊 Multi-Patch Analysis")
                    st.write(f"*Each patch resized to {target_w}×{target_h}*")
                    
                    patches = [
                        ("Center", w//2, h//2),
                        ("Top-Left", w//4, h//4),
                        ("Top-Right", 3*w//4, h//4),
                        ("Bottom-Left", w//4, 3*h//4),
                        ("Bottom-Right", 3*w//4, 3*h//4),
                    ]
                    
                    all_predictions = []
                    half_w, half_h = target_w // 2, target_h // 2
                    
                    for patch_name, x, y in patches:
                        start_x = max(0, x - half_w)
                        end_x = min(w, x + half_w)
                        start_y = max(0, y - half_h)
                        end_y = min(h, y + half_h)
                        
                        if end_x > start_x and end_y > start_y:
                            patch = img_bgr[start_y:end_y, start_x:end_x]
                            patch_resized = cv2.resize(patch, (target_w, target_h))
                            patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
                            
                            patch_processed = np.expand_dims(patch_rgb, axis=0)
                            patch_processed = preprocess_input(patch_processed)
                            
                            pred = model.predict(patch_processed, verbose=0)
                            imagenette_preds = filter_imagenette_predictions(pred[0])
                            
                            if imagenette_preds:
                                top_pred = imagenette_preds[0]
                                all_predictions.append({
                                    'patch': patch_name,
                                    'class': top_pred['class'],
                                    'confidence': top_pred['confidence']
                                })
                    
                    # Display results
                    for pred_info in all_predictions:
                        st.write(f"**{pred_info['patch']} Patch:**")
                        st.write(f"🎯 {pred_info['class']}")
                        st.progress(float(pred_info['confidence']))
                        st.write(f"Confidence: {pred_info['confidence']:.3f} ({pred_info['confidence']*100:.1f}%)")
                        st.write("---")
                    
                    # Summary
                    if all_predictions:
                        class_counts = {}
                        for pred_info in all_predictions:
                            cls = pred_info['class']
                            if cls in class_counts:
                                class_counts[cls] += pred_info['confidence']
                            else:
                                class_counts[cls] = pred_info['confidence']
                        
                        best_class = max(class_counts.keys(), key=lambda k: class_counts[k])
                        avg_conf = class_counts[best_class] / sum(1 for p in all_predictions if p['class'] == best_class)
                        
                        st.subheader("🏆 Overall Prediction")
                        st.success(f"**{best_class}** (Average confidence: {avg_conf:.3f})")
        
        else:
            st.info("👆 Please upload an image to classify")
    
    # Instructions
    st.subheader("📝 How to Use")
    with st.expander("Click to expand instructions"):
        st.markdown("""
        **Imagenette Classes (10 total):**
        1. **tench** (fish)
        2. **English springer** (dog breed)  
        3. **cassette player**
        4. **chain saw**
        5. **church**
        6. **French horn**
        7. **garbage truck**
        8. **gas pump**
        9. **golf ball**
        10. **parachute**
        
        **Best Results:** Upload images containing these specific objects for optimal classification!
        """)

if __name__ == "__main__":
    main()
