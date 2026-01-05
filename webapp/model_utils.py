import onnxruntime
import numpy as np
from PIL import Image
import os

# Define constants
# We'll use the float16 model to fit within Render's memory limits
MODEL_PATH_F16 = os.path.join(os.path.dirname(__file__), '..', 'model_fp16.onnx')
MODEL_PATH_F32 = os.path.join(os.path.dirname(__file__), '..', 'model_float32.onnx')
MODEL_PATH_QUANT = os.path.join(os.path.dirname(__file__), '..', 'model.onnx')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

class RetinalAgeModel:
    def __init__(self, model_path=None):
        if model_path is None:
            if os.path.exists(MODEL_PATH_F16):
                model_path = MODEL_PATH_F16
            elif os.path.exists(MODEL_PATH_F32):
                model_path = MODEL_PATH_F32
            else:
                model_path = MODEL_PATH_QUANT
            
        if os.path.exists(model_path):
            # Load ONNX model
            # Use CPU provider explicitly and optimize for low memory
            options = onnxruntime.SessionOptions()
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=options,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"ONNX Model loaded from {model_path} (Memory Optimized)")
        else:
            print(f"WARNING: Model file not found at {model_path}")
            self.session = None

    def preprocess_image(self, img, laterality):
        """
        Preprocesses the image:
        1. Convert to RGB.
        2. Robust auto-crop to remove black borders.
        3. Handle laterality (flip Left eyes).
        4. Resize to 384x384.
        5. Normalize (ImageNet mean/std).
        """
        img = img.convert('RGB')
        img_np = np.array(img)
        
        # Robust threshold to find fundus area
        # We use a slightly higher threshold (20) to ignore compression noise in black areas
        mask = img_np > 20
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            
            # Add a small padding (2%) to ensure we don't cut the fundus
            h, w = y1 - y0, x1 - x0
            pad_h, pad_w = int(h * 0.02), int(w * 0.02)
            y0 = max(0, y0 - pad_h)
            y1 = min(img_np.shape[0], y1 + pad_h)
            x0 = max(0, x0 - pad_w)
            x1 = min(img_np.shape[1], x1 + pad_w)
            
            img_crop = img_np[y0:y1, x0:x1]
            img = Image.fromarray(img_crop)
        else:
            # Fallback if no mask found
            img = Image.fromarray(img_np)

        # Handle Laterality
        # Model expects disc on the RIGHT (standard for many retinal models)
        # OD (Right) usually has disc on the right in standard fundus images? 
        # Actually, let's stick to the logic that matches the desktop script.
        # Desktop script ensured disc is on the RIGHT.
        # test1.jpg (OD) has disc on the RIGHT.
        # So OD stays as is. OS (disc on left) gets flipped to the right.
        if laterality == 'OS':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize to 384x384
        img = img.resize((384, 384), Image.BICUBIC)
        
        # To Tensor & Normalize
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
        raw_age = output[0].item()
        
        # Apply population-specific recalibration (Age 20-70 subset)
        # Formula: Corrected Age = (0.3582 * Predicted Age) + 33.8793
        recalibrated_age = (0.3582 * raw_age) + 33.8793
            
        return round(recalibrated_age, 1)

# Global instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = RetinalAgeModel()
    return _model_instance
