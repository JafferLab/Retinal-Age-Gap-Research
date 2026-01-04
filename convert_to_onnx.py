import torch
import torch.onnx
from webapp.model_utils import get_model
import onnx
import onnxruntime
import numpy as np

def convert_to_onnx():
    print("Loading PyTorch model...")
    model_wrapper = get_model()
    model = model_wrapper.model
    model.eval()

    # Create dummy input (Batch Size 1, 3 Channels, 384x384)
    dummy_input = torch.randn(1, 3, 384, 384, device=model_wrapper.device)

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
        torch_out = model(dummy_input).cpu().numpy()
    
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print("SUCCESS: ONNX model matches PyTorch model!")

if __name__ == "__main__":
    convert_to_onnx()
