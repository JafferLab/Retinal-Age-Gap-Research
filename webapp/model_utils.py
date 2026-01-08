import onnxruntime
import numpy as np
from PIL import Image
import os

# Define constants
MODEL_PATH_INT8 = os.path.join(os.path.dirname(__file__), 'model_int8.onnx')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Recalibration constants (derived from ODIR-5K 20-70 age analysis)
RECALIB_SLOPE = 0.3582
RECALIB_INTERCEPT = 33.8793

class RetinalAgeModel:
    def __init__(self, model_path=None):
        self.session = None
        self.input_name = None
        
        if model_path is None:
            self.model_path = MODEL_PATH_INT8
        else:
            self.model_path = model_path

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

    def predict(self, img, laterality, recalibration_mode='chinese'):
        """Runs inference on the image.
        
        Args:
            img: PIL Image
            laterality: 'OD' or 'OS'
            recalibration_mode: 'original' or 'chinese'
                - 'chinese': Apply recalibration (default, optimized for ODIR-5K)
                - 'original': Return raw model output (Japanese/JOIR baseline)
        """
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
        
        # The model was trained with recalibration baked in
        # So raw output is already recalibrated for Chinese population
        # To get "original" Japanese output, we need to reverse the recalibration
        if recalibration_mode == 'original':
            # Reverse the recalibration: age = (recalibrated - intercept) / slope
            final_age = (raw_age - RECALIB_INTERCEPT) / RECALIB_SLOPE
        else:
            # Chinese mode: use the model output as-is (already recalibrated)
            final_age = raw_age
            
        return round(final_age, 1)

# Global instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = RetinalAgeModel()
    return _model_instance
