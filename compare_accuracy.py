import torch
import onnxruntime
import numpy as np
from PIL import Image
import os
from webapp.model_utils import RetinalAgeModel as ONNXModel
import timm

# Load original PyTorch model for comparison
def get_pytorch_model():
    model = timm.create_model(model_name='swin_base_patch4_window12_384', num_classes=1, pretrained=False)
    model_path = 'model_20220903.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_pytorch(img):
    # Same logic as in example script
    img_np = np.array(img)
    mask = img_np > 10
    if mask.ndim == 3: mask = mask.any(axis=2)
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = Image.fromarray(img_np[y0:y1, x0:x1])
    
    # For test1, we know it's OD, so no flip needed
    img = img.resize((384, 384), Image.BICUBIC)
    img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0)

def compare():
    test_images = ['test1.jpg', 'test2.jpg']
    
    print("Loading models...")
    pt_model = get_pytorch_model()
    
    # We need the float32 ONNX model too. I'll recreate it if it doesn't exist.
    # Actually, I'll just compare PT vs current Quantized ONNX first.
    quant_model = ONNXModel('model.onnx') # This is the quantized one currently
    
    for img_path in test_images:
        if not os.path.exists(img_path): continue
        print(f"\nTesting {img_path}:")
        img = Image.open(img_path).convert('RGB')
        
        # PyTorch Prediction
        with torch.no_grad():
            pt_input = preprocess_pytorch(img)
            pt_out = pt_model(pt_input).item()
        
        # Quantized ONNX Prediction
        quant_out = quant_model.predict(img, 'OD')
        
        print(f"  PyTorch: {pt_out:.4f}")
        print(f"  Quantized ONNX: {quant_out:.4f}")
        print(f"  Difference: {abs(pt_out - quant_out):.4f}")

if __name__ == "__main__":
    compare()
