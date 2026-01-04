import numpy as np
from PIL import Image
import glob

def check_eye_side(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    # Simple threshold for fundus mask (remove black borders)
    mask = img_np > 10
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img_crop = img_np[y0:y1, x0:x1]
        img = Image.fromarray(img_crop)
    
    # Resize to standard size for consistent processing
    img = img.resize((384, 384), Image.BICUBIC)
    img_gray = np.array(img.convert('L'))
    
    # Threshold to find brightest pixels (optic disc candidate)
    # Use top 2% brightest pixels
    threshold = np.percentile(img_gray, 98)
    bright_mask = img_gray > threshold
    
    # Center of mass of bright pixels
    coords = np.argwhere(bright_mask)
    if coords.size > 0:
        # coords are (y, x)
        mean_y, mean_x = coords.mean(axis=0)
        width = img_gray.shape[1]
        
        is_left_side = mean_x < (width / 2)
        side = "LEFT (Flip needed)" if is_left_side else "RIGHT (No flip)"
        print(f"{img_path}: Optic Disc Center X={mean_x:.1f} (Width={width}) -> {side}")
    else:
        print(f"{img_path}: Could not detect optic disc")

for f in ['test1.jpg', 'test2.jpg', 'test3final.png']:
    check_eye_side(f)
