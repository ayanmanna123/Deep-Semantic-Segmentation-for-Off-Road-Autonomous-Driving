import torch
import torch.onnx
import segmentation_models_pytorch as smp
import os

def convert():
    ENCODER = 'resnet50'
    DEVICE = 'cpu'
    MODEL_PATH = '../../Offroad_Segmentation_Scripts/runs/checkpoints/best_model.pth'
    ONNX_PATH = 'best_model.onnx'
    
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    
    print(f"Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    convert()
