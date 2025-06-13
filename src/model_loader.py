import tensorflow as tf
import streamlit as st
from pathlib import Path
import numpy as np

class MobileNetLoader:
    def __init__(self):
        self.model = None
        self.class_names = []
    
    @st.cache_resource
    def load_mobilenet_model(_self, model_path):
        """Load your custom MobileNetV1 model"""
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading MobileNet model: {e}")
            return None
    
    def set_class_names(self, class_names):
        """Set class names for your model"""
        self.class_names = class_names
    
    def get_model_info(self, model):
        """Get model information"""
        if model is None:
            return None
        
        input_shape = model.input_shape[1:3]  # (height, width)
        num_classes = model.output_shape[-1]
        
        return {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'total_params': model.count_params()
        }
