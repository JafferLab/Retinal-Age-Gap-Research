import torch
import torch.onnx
import timm
import onnx
import onnxruntime
import numpy as np
import os

def convert_to_onnx():
    print("Loading PyTorch model...")
    model = timm.create_model(model_name='swin_base_patch4_window12_384', num_classes=1, pretrained=False)
    model_path = 'model_20220903.pth'
    if not os.path.exists(model_path):
        # Try one level up if not found (for running from webapp/ dir)
        model_path = '../model_20220903.pth'
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input (Batch Size 1, 3 Channels, 384x384)
    dummy_input = torch.randn(1, 3, 384, 384)

    output_path = "model.onnx"
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Compare outputs
    print("Comparing outputs...")
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print("SUCCESS: ONNX model matches PyTorch model!")

if __name__ == "__main__":
    convert_to_onnx()
