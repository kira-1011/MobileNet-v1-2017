import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_loader import MobileNetLoader
from detection_utils import MobileNetDetector

# Page config
st.set_page_config(
    page_title="MobileNetV1 Image Classification",
    page_icon="🖼️",
    layout="wide"
)

def main():
    st.title("🖼️ MobileNetV1 Image Classification")
    st.markdown("*Upload an image and get class predictions from your 32x32 MobileNetV1 model*")
    
    # Sidebar for model and settings
    st.sidebar.title("⚙️ Settings")
    
    # Initialize components
    model_loader = MobileNetLoader()
    
    # Model upload
    st.sidebar.subheader("📁 Load Your Model")
    uploaded_model = st.sidebar.file_uploader(
        "Upload your MobileNetV1 model", 
        type=['keras', 'h5'],
        help="Upload your 32x32 input MobileNetV1 model"
    )
    
    # CIFAR-100 class names preset
    cifar100_classes = """apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle, chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur, dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard, lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain, mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree, plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket, rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider, squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor, train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm"""
    
    use_cifar100 = st.sidebar.checkbox("Use CIFAR-100 class names", value=True)
    
    if use_cifar100:
        class_names_input = cifar100_classes
        st.sidebar.text("Using CIFAR-100 classes")
    else:
        class_names_input = st.sidebar.text_area(
            "Class Names (comma-separated):",
            placeholder="person, car, dog, cat, ...",
            help="Enter the class names your model was trained on"
        )
    
    model = None
    detector = None
    
    # Load model
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(uploaded_model.read())
            temp_path = tmp_file.name
        
        try:
            with st.spinner("Loading your MobileNetV1 model..."):
                model = model_loader.load_mobilenet_model(temp_path)
            
            if model:
                st.sidebar.success("✅ Model loaded successfully!")
                
                # Get model info
                model_info = model_loader.get_model_info(model)
                st.sidebar.write(f"**Input Size:** {model_info['input_shape']}")
                st.sidebar.write(f"**Classes:** {model_info['num_classes']}")
                st.sidebar.write(f"**Parameters:** {model_info['total_params']:,}")
                
                # Process class names
                if class_names_input.strip():
                    if use_cifar100:
                        class_names = [name.strip() for name in class_names_input.split(',')]
                    else:
                        class_names = [name.strip() for name in class_names_input.replace('\n', ',').split(',') if name.strip()]
                else:
                    class_names = [f"Class_{i}" for i in range(model_info['num_classes'])]
                
                # Trim class names to match model output
                class_names = class_names[:model_info['num_classes']]
                
                # Create detector
                detector = MobileNetDetector(model, class_names)
                
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
        
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    # Prediction settings
    if detector:
        st.sidebar.subheader("🎯 Prediction Settings")
        
        prediction_mode = st.sidebar.selectbox(
            "Prediction Mode:",
            ["whole_image", "center_patch", "multiple_patches"],
            help="How to analyze the image"
        )
        
        num_predictions = st.sidebar.slider(
            "Number of Top Predictions", 1, 10, 5, 1,
            help="Show top N predictions"
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify"
        )
        
        if uploaded_image:
            # Display original image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to numpy array and handle different formats
            img_array = np.array(image)
            
            # Handle different image formats
            if len(img_array.shape) == 2:
                # Grayscale - convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    # RGBA - remove alpha channel
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 3:
                    # RGB - perfect
                    pass
                else:
                    st.error(f"Unsupported image format: {img_array.shape[2]} channels")
                    return
            else:
                st.error("Invalid image format")
                return
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.write(f"**Image Size:** {img_array.shape[1]} x {img_array.shape[0]} pixels")
            st.write(f"**Channels:** {img_array.shape[2] if len(img_array.shape) == 3 else 1}")
            
    with col2:
        st.subheader("🎯 Classification Results")
        
        if uploaded_image and detector:
            with st.spinner("Analyzing image..."):
                
                h, w = img_bgr.shape[:2]
                
                if prediction_mode == "whole_image":
                    # Resize entire image to 32x32
                    resized_img = cv2.resize(img_bgr, (32, 32))
                    img_processed = np.expand_dims(resized_img.astype(np.float32) / 255.0, axis=0)
                    
                    pred = detector.model.predict(img_processed, verbose=0)
                    
                    st.subheader("📊 Whole Image Classification")
                    
                    # Show top N predictions
                    top_indices = np.argsort(pred[0])[-num_predictions:][::-1]
                    
                    for i, idx in enumerate(top_indices, 1):
                        conf = pred[0][idx]
                        class_name = detector.class_names[idx]
                        
                        # Create a progress bar for confidence
                        st.write(f"**{i}. {class_name}**")
                        st.progress(float(conf))
                        st.write(f"Confidence: {conf:.3f} ({conf*100:.1f}%)")
                        st.write("---")
                
                elif prediction_mode == "center_patch":
                    # Get center 32x32 patch
                    center_x, center_y = w//2, h//2
                    patch = img_bgr[center_y-16:center_y+16, center_x-16:center_x+16]
                    
                    if patch.shape[0] == 32 and patch.shape[1] == 32:
                        patch_processed = np.expand_dims(patch.astype(np.float32) / 255.0, axis=0)
                        pred = detector.model.predict(patch_processed, verbose=0)
                        
                        st.subheader("📊 Center Patch Classification")
                        
                        # Show top N predictions
                        top_indices = np.argsort(pred[0])[-num_predictions:][::-1]
                        
                        for i, idx in enumerate(top_indices, 1):
                            conf = pred[0][idx]
                            class_name = detector.class_names[idx]
                            
                            st.write(f"**{i}. {class_name}**")
                            st.progress(float(conf))
                            st.write(f"Confidence: {conf:.3f} ({conf*100:.1f}%)")
                            st.write("---")
                    else:
                        st.error("Image too small for center patch analysis")
                
                elif prediction_mode == "multiple_patches":
                    # Analyze multiple patches from different areas
                    st.subheader("📊 Multi-Patch Analysis")
                    
                    patches = [
                        ("Center", w//2, h//2),
                        ("Top-Left", w//4, h//4),
                        ("Top-Right", 3*w//4, h//4),
                        ("Bottom-Left", w//4, 3*h//4),
                        ("Bottom-Right", 3*w//4, 3*h//4),
                    ]
                    
                    all_predictions = []
                    
                    for patch_name, x, y in patches:
                        if x >= 16 and y >= 16 and x < w-16 and y < h-16:
                            patch = img_bgr[y-16:y+16, x-16:x+16]
                            patch_processed = np.expand_dims(patch.astype(np.float32) / 255.0, axis=0)
                            pred = detector.model.predict(patch_processed, verbose=0)
                            
                            top_class_idx = np.argmax(pred[0])
                            top_conf = np.max(pred[0])
                            top_class = detector.class_names[top_class_idx]
                            
                            all_predictions.append({
                                'patch': patch_name,
                                'class': top_class,
                                'confidence': top_conf
                            })
                    
                    # Display results
                    for pred_info in all_predictions:
                        st.write(f"**{pred_info['patch']} Patch:**")
                        st.write(f"🎯 {pred_info['class']}")
                        st.progress(float(pred_info['confidence']))
                        st.write(f"Confidence: {pred_info['confidence']:.3f} ({pred_info['confidence']*100:.1f}%)")
                        st.write("---")
                    
                    # Summary - most common prediction
                    class_counts = {}
                    for pred_info in all_predictions:
                        cls = pred_info['class']
                        if cls in class_counts:
                            class_counts[cls] += pred_info['confidence']
                        else:
                            class_counts[cls] = pred_info['confidence']
                    
                    if class_counts:
                        best_class = max(class_counts.keys(), key=lambda k: class_counts[k])
                        avg_conf = class_counts[best_class] / sum(1 for p in all_predictions if p['class'] == best_class)
                        
                        st.subheader("🏆 Overall Prediction")
                        st.success(f"**{best_class}** (Average confidence: {avg_conf:.3f})")
        
        elif uploaded_image and not detector:
            st.info("👆 Please upload your MobileNetV1 model first")
        
        elif not uploaded_image and detector:
            st.info("👆 Please upload an image to classify")
        
        else:
            st.info("👆 Upload both a model and an image to start classification")
    
    # Instructions
    st.subheader("📝 How to Use")
    with st.expander("Click to expand instructions"):
        st.markdown("""
        1. **Upload your MobileNetV1 model** (.keras or .h5 file) in the sidebar
        2. **Check "Use CIFAR-100 class names"** if your model was trained on CIFAR-100
        3. **Upload an image** using the file uploader
        4. **Choose prediction mode**:
           - **Whole Image**: Resize entire image to 32x32 and classify
           - **Center Patch**: Take 32x32 patch from center and classify
           - **Multiple Patches**: Analyze 5 different 32x32 patches
        5. **View top predictions** with confidence scores and progress bars
        
        **Tips:**
        - **Whole Image** works best for simple, centered objects
        - **Center Patch** is good when the main object is in the center
        - **Multiple Patches** gives you more comprehensive analysis
        - Try images with objects similar to CIFAR-100 classes
        """)

if __name__ == "__main__":
    main()
