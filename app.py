import streamlit as st
import cv2
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_loader import MobileNetLoader
from detection_utils import MobileNetDetector
from camera_utils import MobileNetVideoTransformer, get_webrtc_config
from streamlit_webrtc import webrtc_streamer

# Page config
st.set_page_config(
    page_title="MobileNetV1 32x32 Real-Time Detection",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 MobileNetV1 (32x32) Real-Time Object Detection")
    st.markdown("*Optimized for 32x32 input models (e.g., CIFAR-100 trained)*")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Initialize components
    model_loader = MobileNetLoader()
    
    # Model upload
    st.sidebar.subheader("📁 Load Your 32x32 Model")
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
    
    if uploaded_model:
        # Save uploaded file temporarily
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
                
                # Verify input size
                if model_info['input_shape'] != (32, 32):
                    st.sidebar.warning(f"⚠️ Model input is {model_info['input_shape']}, expected (32, 32)")
                
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
    
    # Detection settings
    st.sidebar.subheader("🎯 Detection Settings")
    
    detection_method = st.sidebar.selectbox(
        "Detection Method:",
        ["adaptive", "sliding_window", "grid"],
        help="Adaptive automatically chooses the best method based on image size"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.6, 0.05
    )
    
    if detection_method == "sliding_window":
        stride = st.sidebar.slider(
            "Stride (pixels)", 8, 32, 16, 2,
            help="Smaller = more thorough but slower"
        )
    else:
        stride = 16
    
    # Performance settings
    frame_skip = st.sidebar.slider(
        "Frame Skip", 1, 5, 2, 1,
        help="Process every Nth frame (higher = faster but less smooth)"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        if detector:
            # Initialize video transformer
            video_transformer = MobileNetVideoTransformer()
            video_transformer.set_detector(detector)
            video_transformer.set_parameters(detection_method, confidence_threshold, stride, frame_skip)
            
            # WebRTC streamer
            ctx = webrtc_streamer(
                key="mobilenet-32x32-detection",
                video_transformer_factory=lambda: video_transformer,
                rtc_configuration=get_webrtc_config(),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.info("👆 Please upload your 32x32 MobileNetV1 model to start detection")
            
            # Show model requirements
            st.subheader("📋 Model Requirements")
            st.markdown("""
            Your model should have:
            - **Input shape**: (32, 32, 3)
            - **Output**: Number of classes you want to detect
            - **Training**: Preferably on CIFAR-100 or similar dataset
            
            Example model creation:
            ```python
            model = MobileNetV1(alpha=1.0, input_shape=(32, 32, 3), num_classes=100)
            ```
            """)
    
    with col2:
        st.subheader("ℹ️ Detection Info")
        
        if detector:
            st.write(f"**Model:** MobileNetV1 (32x32)")
            st.write(f"**Method:** {detection_method.replace('_', ' ').title()}")
            st.write(f"**Classes:** {len(detector.class_names)}")
            st.write(f"**Confidence:** {confidence_threshold}")
            st.write(f"**Frame Skip:** {frame_skip}")
            
            if detection_method == "sliding_window":
                st.write(f"**Stride:** {stride}px")
            
            # Show some class names
            with st.expander("View Classes (first 20)"):
                for i, class_name in enumerate(detector.class_names[:20]):
                    st.write(f"{i}: {class_name}")
                if len(detector.class_names) > 20:
                    st.write(f"... and {len(detector.class_names) - 20} more")
        
        # Method explanations for 32x32
        st.subheader("🔍 Detection Methods")
        
        with st.expander("Adaptive (Recommended)"):
            st.markdown("""
            - **Automatically chooses** best method based on camera resolution
            - Small windows → Grid detection
            - Medium windows → Sliding window  
            - Large windows → Pyramid detection
            """)
        
        with st.expander("Sliding Window"):
            st.markdown("""
            - Moves 32x32 window across image
            - Multiple scales for different object sizes
            - **Best for:** Medium-sized objects
            """)
        
        with st.expander("Grid Detection"):
            st.markdown("""
            - Divides image into overlapping grid
            - Each cell resized to 32x32
            - **Best for:** Fast detection, small objects
            """)
        
        # Performance tips for 32x32
        st.subheader("⚡ 32x32 Optimization Tips")
        st.markdown("""
        - **Small objects** work best (32x32 is small!)
        - **Higher frame skip** for better performance
        - **Grid method** is fastest
        - **Good contrast** helps with small details
        - **Objects 50-200px** in camera work well
        """)

if __name__ == "__main__":
    main()