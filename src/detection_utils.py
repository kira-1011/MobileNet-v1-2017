import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_ubyte

class MobileNetDetector:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or [f"Class_{i}" for i in range(model.output_shape[-1])]
        self.input_size = model.input_shape[1:3]  # Should be (32, 32)
        print(f"Model input size: {self.input_size}")
    
    def sliding_window_detection(self, image, stride=16, confidence_threshold=0.7, scale_factor=1.5):
        """Optimized sliding window for 32x32 model"""
        detections = []
        h, w = image.shape[:2]
        window_h, window_w = self.input_size  # (32, 32)
        
        # Multi-scale detection for better results with small windows
        scales = [1.0, scale_factor, scale_factor**2] if scale_factor > 1 else [1.0]
        
        for scale in scales:
            # Scale the window size
            scaled_window_h = int(window_h * scale)
            scaled_window_w = int(window_w * scale)
            
            # Skip if scaled window is larger than image
            if scaled_window_h > h or scaled_window_w > w:
                continue
            
            for y in range(0, h - scaled_window_h + 1, stride):
                for x in range(0, w - scaled_window_w + 1, stride):
                    # Extract window at current scale
                    window = image[y:y+scaled_window_h, x:x+scaled_window_w]
                    
                    # Resize to model input size (32x32)
                    window_resized = cv2.resize(window, (window_w, window_h))
                    
                    # Preprocess for model
                    window_processed = self.preprocess_image(window_resized)
                    
                    # Predict
                    pred = self.model.predict(window_processed, verbose=0)
                    confidence = np.max(pred)
                    class_id = np.argmax(pred)
                    
                    if confidence > confidence_threshold:
                        detections.append({
                            'bbox': [x, y, scaled_window_w, scaled_window_h],
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': self.class_names[class_id],
                            'scale': scale
                        })
        
        # Apply Non-Maximum Suppression
        detections = self.apply_nms(detections, iou_threshold=0.3)
        return detections
    
    def grid_detection(self, image, grid_size=8, overlap=0.5, confidence_threshold=0.7):
        """Grid-based detection optimized for 32x32 input"""
        detections = []
        h, w = image.shape[:2]
        
        # Calculate step size with overlap
        step_h = int(h / grid_size * (1 - overlap))
        step_w = int(w / grid_size * (1 - overlap))
        
        # Calculate actual window sizes to cover the image
        window_h = h // grid_size + int(h / grid_size * overlap)
        window_w = w // grid_size + int(w / grid_size * overlap)
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate window position
                y = min(i * step_h, h - window_h)
                x = min(j * step_w, w - window_w)
                
                # Extract window
                window = image[y:y+window_h, x:x+window_w]
                
                # Resize to 32x32
                window_resized = cv2.resize(window, self.input_size)
                window_processed = self.preprocess_image(window_resized)
                
                # Predict
                pred = self.model.predict(window_processed, verbose=0)
                confidence = np.max(pred)
                class_id = np.argmax(pred)
                
                if confidence > confidence_threshold:
                    detections.append({
                        'bbox': [x, y, window_w, window_h],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id],
                        'grid_pos': (i, j)
                    })
        
        return detections
    
    def pyramid_detection(self, image, confidence_threshold=0.7):
        """Image pyramid detection for multi-scale objects"""
        detections = []
        
        # Create image pyramid
        pyramid_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        min_size = 32  # Don't go smaller than model input
        
        for scale in pyramid_scales:
            # Scale image
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            
            # Skip if too small
            if new_h < min_size or new_w < min_size:
                continue
            
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # Apply sliding window on scaled image
            stride = max(8, int(16 * scale))  # Adaptive stride
            scale_detections = self.sliding_window_detection(
                scaled_image, stride=stride, 
                confidence_threshold=confidence_threshold,
                scale_factor=1.0  # No additional scaling in sliding window
            )
            
            # Scale back detections to original image size
            for det in scale_detections:
                x, y, w, h = det['bbox']
                det['bbox'] = [
                    int(x / scale), int(y / scale),
                    int(w / scale), int(h / scale)
                ]
                det['pyramid_scale'] = scale
            
            detections.extend(scale_detections)
        
        # Apply NMS across all scales
        detections = self.apply_nms(detections, iou_threshold=0.4)
        return detections
    
    def adaptive_detection(self, image, confidence_threshold=0.7):
        """Adaptive detection that chooses method based on image size"""
        h, w = image.shape[:2]
        
        # For small images, use grid detection
        if h <= 320 and w <= 320:
            return self.grid_detection(image, grid_size=6, confidence_threshold=confidence_threshold)
        
        # For medium images, use sliding window
        elif h <= 640 and w <= 640:
            return self.sliding_window_detection(image, stride=12, confidence_threshold=confidence_threshold)
        
        # For large images, use pyramid detection
        else:
            return self.pyramid_detection(image, confidence_threshold=confidence_threshold)
    
    def preprocess_image(self, image):
        """Preprocess image for 32x32 MobileNet model"""
        # Ensure correct size
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1] (adjust based on your training preprocessing)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression with lower threshold for 32x32"""
        if not detections:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            keep_detection = True
            
            for kept_det in keep:
                iou = self.calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    keep_detection = False
                    break
            
            if keep_detection:
                keep.append(det)
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels optimized for small detection windows"""
        for det in detections:
            x, y, w, h = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Use different colors for different scales if available
            if 'scale' in det:
                scale = det['scale']
                if scale <= 1.0:
                    color = (0, 255, 0)    # Green for small scale
                elif scale <= 1.5:
                    color = (255, 255, 0)  # Yellow for medium scale
                else:
                    color = (0, 255, 255)  # Cyan for large scale
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box with thicker lines for small boxes
            thickness = 3 if (w < 50 or h < 50) else 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with adaptive text size
            text_size = 0.5 if (w < 80 or h < 80) else 0.7
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text dimensions
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2
            )
            
            # Position label above box if there's space, otherwise inside
            label_y = y - 5 if y > text_height + 10 else y + text_height + 5
            
            # Label background
            cv2.rectangle(image, (x, label_y - text_height - 5), 
                         (x + text_width, label_y + 5), color, -1)
            
            # Label text
            cv2.putText(image, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)
        
        return image
