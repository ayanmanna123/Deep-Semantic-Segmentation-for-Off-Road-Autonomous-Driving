import cv2
import torch
import numpy as np
import os
import segmentation_models_pytorch as smp
import albumentations as A
from path_planner import PathPlanner

def main():
    # --- CONFIGURATION ---
    MODEL_PATH = './runs/checkpoints/best_model.pth'
    ENCODER = 'timm-efficientnet-b3' 
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Appearance
    ALPHA = 0.4 # Transparency for segmentation overlay
    
    # --- INITIALIZE COMPONENTS ---
    print(f"Loading model on {DEVICE}...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        classes=CLASSES, 
        activation=None
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train a model first.")
        return

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Setup preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    planner = PathPlanner()
    
    # Palette matches PathPlanner and training
    PALETTE = np.array([
        [34, 139, 34],    # 0: Trees - Forest Green
        [0, 255, 0],      # 1: Lush Bushes - Lime
        [210, 180, 140],  # 2: Dry Grass - Tan
        [139, 90, 43],    # 3: Dry Bushes - Brown
        [128, 128, 0],    # 4: Ground Clutter - Olive
        [255, 105, 180],  # 5: Flowers - Pink
        [139, 69, 19],    # 6: Logs - Brown
        [128, 128, 128],  # 7: Rocks - Gray
        [160, 82, 45],    # 8: Landscape - Sienna
        [135, 206, 235]   # 9: Sky - Sky Blue
    ], dtype=np.uint8)

    # --- START CAMERA ---
    cap = cv2.VideoCapture(0) # Use 0 for primary webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\n--- Real-Time Off-Road Simulation Started ---")
    print("Press 'q' to exit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Prepare Image
            # Resize for model processing (keeps it fast)
            h_orig, w_orig = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(frame_rgb, (320, 320))
            
            # Preprocess
            input_pre = preprocessing_fn(input_img).astype('float32')
            input_tensor = torch.from_numpy(input_pre.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            # 2. Inference (Semantic Segmentation)
            output = model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # 3. Path Planning (A*)
            # The planner handles internal scaling
            path = planner.find_safest_path(mask)
            
            # Scale path back to original resolution for display
            scale_h = h_orig / 320
            scale_w = w_orig / 320
            scaled_path = [(int(x * scale_w), int(y * scale_h)) for x, y in path]

            # 4. Visualization
            # Create color mask overlay
            mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            color_mask = PALETTE[mask_resized]
            
            # Blend frame and mask
            overlay = cv2.addWeighted(frame, 1 - ALPHA, color_mask[:, :, ::-1], ALPHA, 0) # Convert RGB to BGR for CV2
            
            # Draw Path
            final_frame = planner.visualize_on_image(overlay, scaled_path)

            # UI Text
            cv2.putText(final_frame, "AI OFF-ROAD NAVIGATION", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(final_frame, f"Device: {DEVICE}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Display
            cv2.imshow('Real-Time OFF-ROAD AI Simulator', final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
