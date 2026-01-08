import onnxruntime
import numpy as np
from PIL import Image
import os

# Define constants
# We'll use the int8 model to fit within Render's strict memory limits
MODEL_PATH_INT8 = os.path.join(os.path.dirname(__file__), 'model_int8.onnx')
MODEL_PATH_F16 = os.path.join(os.path.dirname(__file__), '..', 'model_fp16.onnx')
MODEL_PATH_F32 = os.path.join(os.path.dirname(__file__), '..', 'model_float32.onnx')
MODEL_PATH_QUANT = os.path.join(os.path.dirname(__file__), '..', 'model.onnx')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

class RetinalAgeModel:
    def __init__(self, model_path=None):
        self.session = None
        self.input_name = None
        self.model_path = model_path
        
        if self.model_path is None:
            # Priority: int8 (smallest) -> fp16 -> fp32
            if os.path.exists(MODEL_PATH_INT8):
                self.model_path = MODEL_PATH_INT8
            elif os.path.exists(MODEL_PATH_F16):
                self.model_path = MODEL_PATH_F16
            elif os.path.exists(MODEL_PATH_F32):
                self.model_path = MODEL_PATH_F32
            else:
                self.model_path = MODEL_PATH_QUANT

    def _load_model(self):
        """Lazy load the ONNX session."""
        if self.session is not None:
            return

        if os.path.exists(self.model_path):
            # Load ONNX model with memory optimizations
            options = onnxruntime.SessionOptions()
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                sess_options=options,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"ONNX Model loaded from {self.model_path} (Memory Optimized)")
        else:
            raise RuntimeError(f"Model file not found at {self.model_path}")

    def preprocess_image(self, img, laterality):
        img = img.convert('RGB')
        img_np = np.array(img)
        
        mask = img_np > 20
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            h, w = y1 - y0, x1 - x0
            pad_h, pad_w = int(h * 0.02), int(w * 0.02)
            y0, y1 = max(0, y0 - pad_h), min(img_np.shape[0], y1 + pad_h)
            x0, x1 = max(0, x0 - pad_w), min(img_np.shape[1], x1 + pad_w)
            img = Image.fromarray(img_np[y0:y1, x0:x1])
        else:
            img = Image.fromarray(img_np)

        if laterality == 'OS':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        img = img.resize((384, 384), Image.BICUBIC)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_np = (img_np - MEAN) / STD
        return img_np

    def predict(self, img, laterality, recalibration_mode='original'):
        """Runs inference on the image."""
        import gc
        self._load_model() # Ensure model is loaded
        
        processed_img = self.preprocess_image(img, laterality)
        img_batch = np.expand_dims(processed_img, axis=0).astype(np.float32)
        
        output = self.session.run(None, {self.input_name: img_batch})
        raw_age = output[0].item()
        
        # Explicitly delete large objects and trigger GC
        del processed_img
        del img_batch
        gc.collect()
        
        # Apply population-specific recalibration if requested
        if recalibration_mode == 'chinese':
            # Formula: Corrected Age = (0.3582 * Predicted Age) + 33.8793
            final_age = (0.3582 * raw_age) + 33.8793
        else:
            # Original JOIR model output
            final_age = raw_age
            
        return round(final_age, 1)

# Global instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = RetinalAgeModel()
    return _model_instance
