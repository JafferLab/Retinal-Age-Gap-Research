import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

# Define constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_20220903.pth')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class RetinalAgeModel:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name='swin_base_patch4_window12_384', num_classes=1, pretrained=False)
        self.model.to(self.device)
        
        if model_path is None:
            model_path = MODEL_PATH
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path} on {self.device}")
        else:
            print(f"WARNING: Model file not found at {model_path}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def preprocess_image(self, img, laterality):
        """
        Preprocesses the image:
        1. Convert to RGB.
        2. Auto-crop to remove black borders.
        3. Handle laterality (flip Left eyes).
        4. Resize to 384x384.
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
        # If user specifies 'OS' (Left), we flip it to look like 'OD' (Right)
        # The model is trained on OD (or flipped OS) images.
        if laterality == 'OS':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize to 384x384
        img = img.resize((384, 384), Image.BICUBIC)
        return img

    def predict(self, img, laterality):
        """
        Runs inference on the image.
        """
        processed_img = self.preprocess_image(img, laterality)
        img_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            age = output.item()
            
        return round(age, 1)

# Global instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = RetinalAgeModel()
    return _model_instance
