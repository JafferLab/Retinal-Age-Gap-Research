import onnxruntime
import numpy as np
from PIL import Image
import os

# Define constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.onnx')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

class RetinalAgeModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
            
        if os.path.exists(model_path):
            # Load ONNX model
            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            print(f"ONNX Model loaded from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path}")
            self.session = None

    def preprocess_image(self, img, laterality):
        """
        Preprocesses the image:
        1. Convert to RGB.
        2. Auto-crop to remove black borders.
        3. Handle laterality (flip Left eyes).
        4. Resize to 384x384.
        5. Normalize (ImageNet mean/std).
        """
        img = img.convert('RGB')
        img_np = np.array(img)
        
        # Threshold to find fundus area (remove black borders)
        mask = img_np > 10
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            img_crop = img_np[y0:y1, x0:x1]
            img = Image.fromarray(img_crop)
        else:
            # Fallback if no mask found
            img = Image.fromarray(img_np)

        # Handle Laterality
        if laterality == 'OS':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize to 384x384
        img = img.resize((384, 384), Image.BICUBIC)
        
        # To Tensor & Normalize (Manual implementation for Numpy)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1) # HWC -> CHW
        img_np = (img_np - MEAN) / STD
        
        return img_np

    def predict(self, img, laterality):
        """
        Runs inference on the image using ONNX Runtime.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        processed_img = self.preprocess_image(img, laterality)
        
        # Add batch dimension
        img_batch = np.expand_dims(processed_img, axis=0).astype(np.float32)
        
        # Run inference
        output = self.session.run(None, {self.input_name: img_batch})
        age = output[0].item()
            
        return round(age, 1)

# Global instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = RetinalAgeModel()
    return _model_instance
