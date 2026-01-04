import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_model():
    input_model = "model.onnx"
    output_model = "model_quantized.onnx"
    
    if not os.path.exists(input_model):
        print(f"Error: {input_model} not found.")
        return

    print(f"Quantizing {input_model} to {output_model}...")
    
    # Dynamic quantization is easiest and often effective for regression models
    quantize_dynamic(
        input_model,
        output_model,
        weight_type=QuantType.QUInt8
    )
    
    print("Quantization complete.")
    
    # Compare sizes
    old_size = os.path.getsize(input_model) / (1024 * 1024)
    new_size = os.path.getsize(output_model) / (1024 * 1024)
    print(f"Original size: {old_size:.2f} MB")
    print(f"Quantized size: {new_size:.2f} MB")
    print(f"Reduction: {((old_size - new_size) / old_size) * 100:.1f}%")

if __name__ == "__main__":
    quantize_model()
