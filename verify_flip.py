from PIL import Image
import torch
from example_GPU_20230309 import preprocess_image, model, device, transform

# Create a fake Left Eye image (flip test1)
img = Image.open('test1.jpg').convert('RGB')
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flipped.save('test1_flipped.jpg')

print("Created test1_flipped.jpg (Simulated Left Eye)")

# Run inference on the flipped image
img_processed = preprocess_image(img_flipped)
img_tensor = transform(img_processed).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    pred = output.item()

print(f"Prediction for test1_flipped.jpg: {pred:.2f}")
print(f"Expected (approx): 49.74")
