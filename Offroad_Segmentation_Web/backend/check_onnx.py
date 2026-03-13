import onnxruntime as ort
import numpy as np

try:
    session = ort.InferenceSession('best_model.onnx')
    print("ONNX model loaded successfully!")
    dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
    outputs = session.run(None, {'input': dummy_input})
    print("ONNX inference successful! Output shape:", outputs[0].shape)
except Exception as e:
    print(f"Error loading/running ONNX model: {e}")
