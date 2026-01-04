import numpy as np
from PIL import Image, ImageFilter

def check_quality(img):
    """
    Checks image quality for:
    1. Field Adequacy (Ratio of non-black pixels)
    2. Exposure (Mean intensity)
    3. Blur (Variance of Laplacian)
    
    Returns:
        status: 'PASS', 'WARN', 'FAIL'
        metrics: dict of measured values
    """
    img = img.convert('RGB')
    img_np = np.array(img)
    img_gray = np.array(img.convert('L'))
    
    # 1. Field Adequacy
    # Count non-black pixels (threshold > 10)
    mask = img_gray > 10
    total_pixels = img_gray.size
    valid_pixels = np.count_nonzero(mask)
    field_ratio = valid_pixels / total_pixels
    
    # 2. Exposure
    # Mean intensity of valid pixels
    if valid_pixels > 0:
        mean_intensity = np.mean(img_gray[mask])
    else:
        mean_intensity = 0
        
    # 3. Blur
    # Laplacian Variance
    # Use PIL's Kernel filter to approximate Laplacian
    # Kernel for Laplacian: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    laplacian_kernel = ImageFilter.Kernel((3, 3), [0, -1, 0, -1, 4, -1, 0, -1, 0], scale=1)
    edges = img.convert('L').filter(laplacian_kernel)
    edges_np = np.array(edges)
    # Calculate variance only on valid fundus area to avoid black background affecting score
    if valid_pixels > 0:
        blur_score = np.var(edges_np[mask])
    else:
        blur_score = 0
        
    metrics = {
        'field_ratio': round(field_ratio, 3),
        'mean_intensity': round(mean_intensity, 1),
        'blur_score': round(blur_score, 1)
    }
    
    # Thresholds (Heuristic)
    status = 'PASS'
    reasons = []
    
    # Field Adequacy Threshold
    if field_ratio < 0.4: # Fundus usually takes up significant space
        status = 'FAIL'
        reasons.append('Field too small')
        
    # Exposure Thresholds
    if mean_intensity < 30:
        status = 'FAIL'
        reasons.append('Underexposed')
    elif mean_intensity > 200:
        status = 'WARN'
        reasons.append('Overexposed')
        
    # Blur Threshold
    # Low variance = blurry
    if blur_score < 50: # Adjust based on testing
        if status != 'FAIL': status = 'WARN'
        reasons.append('Blurry')
        
    return {
        'status': status,
        'metrics': metrics,
        'reasons': reasons
    }
