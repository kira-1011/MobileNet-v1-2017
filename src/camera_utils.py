import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np

class MobileNetVideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.detector = None
        self.detection_method = "adaptive"
        self.confidence_threshold = 0.7
        self.stride = 16  # Smaller stride for 32x32
        self.frame_skip = 2  # Process every N frames for performance
        self.frame_count = 0
        self.last_detections = []
    
    def set_detector(self, detector):
        self.detector = detector
    
    def set_parameters(self, method, confidence, stride, frame_skip=2):
        self.detection_method = method
        self.confidence_threshold = confidence
        self.stride = stride
        self.frame_skip = frame_skip
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        print(f"🎬 Frame {self.frame_count}: Processing...")
        
        if self.detector is not None:
            # SIMPLE TEST: Just predict center patch every frame
            h, w = img.shape[:2]
            center_x, center_y = w//2, h//2
            
            # Get 32x32 patch from center
            patch = img[center_y-16:center_y+16, center_x-16:center_x+16]
            patch = cv2.resize(patch, (32, 32))  # Ensure 32x32
            patch_processed = np.expand_dims(patch.astype(np.float32) / 255.0, axis=0)
            
            # Predict
            pred = self.detector.model.predict(patch_processed, verbose=0)
            top_conf = np.max(pred)
            top_class = np.argmax(pred)
            top_name = self.detector.class_names[top_class]
            
            print(f"🧠 CENTER PREDICTION: {top_name} = {top_conf:.3f}")
            
            # Draw a box in center with prediction
            cv2.rectangle(img, (center_x-50, center_y-50), (center_x+50, center_y+50), (0, 255, 0), 2)
            cv2.putText(img, f"{top_name}: {top_conf:.2f}", (center_x-50, center_y-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"📦 DRAWING BOX: ({center_x-50}, {center_y-50}) to ({center_x+50}, {center_y+50})")
        else:
            print("❌ No detector loaded")
            # No model loaded message
            cv2.putText(img, "Upload your 32x32 MobileNetV1 model", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_webrtc_config():
    """Get WebRTC configuration"""
    return RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
