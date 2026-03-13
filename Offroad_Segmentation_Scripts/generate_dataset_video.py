import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
import sys
import time

# Add backend to path for imports
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Offroad_Segmentation_Web', 'backend'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from path_planner import PathPlanner
from astar_planner import AStarPlanner
from model_utils import mask_to_rgb, get_overlay

def create_demo_video():
    # --- CONFIG ---
    IMAGES_DIR = r'C:\Users\manna\Coding\Hackthons\Deep-Semantic-Segmentation-for-Off-Road-Autonomous-Driving\Offroad_Segmentation_Training_Dataset\train\Color_Images'
    # Use the same best_model.onnx that the server uses for consistency if possible, 
    # but the user might want the .pth weights path. Let's use the one from visualize_final.py
    MODEL_PATH = r'C:\Users\manna\Coding\Hackthons\Deep-Semantic-Segmentation-for-Off-Road-Autonomous-Driving\runs\checkpoints\best_model (1).pth'
    OUTPUT_VIDEO = 'offroad_path_demonstration.mp4'
    ENCODER = 'resnet50' # Standard for this project based on model_utils
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FPS = 4
    
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    planner = AStarPlanner() # Using AStarPlanner as it's the latest used in real-time server
    
    # 1. Select images 12-412
    image_ids = []
    for i in range(12, 413):
        # Format cc0000XXX.png
        filename = f"cc{str(i).zfill(7)}.png"
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            image_ids.append(path)
    
    if not image_ids:
        print(f"No images found in {IMAGES_DIR} with pattern cc0000XXX.png in range 12-412")
        return

    print(f"Found {len(image_ids)} images. Generating video...")
    
    # Initialize video writer
    first_img = cv2.imread(image_ids[0])
    h_orig, w_orig = first_img.shape[:2]
    # We will show side-by-side or just the overlay? 
    # User said "show realtime path", usually means overlay or side-by-side. 
    # Let's do side-by-side like visualize_final.py for comparison.
    combined_w = w_orig * 2 + 10 # 10 is divider
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (combined_w, h_orig))

    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_ids)):
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Prepare image for model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_size = 320
            image_resized = cv2.resize(image_rgb, (input_size, input_size))
            preprocessed = preprocessing_fn(image_resized).astype(np.float32)
            input_tensor = torch.from_numpy(preprocessed.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # Inference
            output = model(input_tensor)
            mask_indices = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # 2. Run A* Pathfinding (on model size 320x320, optionally downsampled for stability)
            # Real-time server uses DS_FACTOR 2 (160x160)
            DS_FACTOR = 2
            mask_ds = cv2.resize(mask_indices, (input_size // DS_FACTOR, input_size // DS_FACTOR), interpolation=cv2.INTER_NEAREST)
            path_ds = planner.find_path(mask_ds)
            
            # Scale path back to original image size
            scale_x = w_orig / (input_size // DS_FACTOR)
            scale_y = h_orig / (input_size // DS_FACTOR)
            scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path_ds]
            
            # 3. Create Visualization (Thick Blue Ribbon)
            mask_rgb = mask_to_rgb(mask_indices)
            mask_rgb_resized = cv2.resize(mask_rgb, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # Switch mask_rgb to BGR for OpenCV
            mask_bgr = cv2.cvtColor(mask_rgb_resized, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(image, 0.5, mask_bgr, 0.5, 0)
            
            if len(scaled_path) > 1:
                pts = np.array(scaled_path, np.int32).reshape((-1, 1, 2))
                
                # Create a blue path effect (Thick Ribbon)
                path_mask = np.zeros_like(overlay)
                # Color (255, 80, 0) is Blue in BGR
                cv2.polylines(path_mask, [pts], isClosed=False, color=(255, 80, 0), thickness=30) 
                
                # Blend the path onto the overlay
                path_indices = np.where(np.any(path_mask > 0, axis=-1))
                overlay[path_indices] = cv2.addWeighted(overlay[path_indices], 0.2, path_mask[path_indices], 0.8, 0)
                
                # Add a sharper center line (Light Blue)
                cv2.polylines(overlay, [pts], isClosed=False, color=(255, 200, 100), thickness=4)

            # Concatenate side-by-side
            divider = np.ones((h_orig, 10, 3), dtype=np.uint8) * 200
            combined = np.hstack([image, divider, overlay])
            
            # Add text info
            cv2.putText(combined, f"Frame: {i+1}/{len(image_ids)} (Photo {os.path.basename(img_path)})", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Realtime Path Planner Demo - 4 Hz", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(combined)
            
            # Optional: Show window if not running headless
            # cv2.imshow("Offroad Path Demo", combined)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    out.release()
    print(f"Video saved as {OUTPUT_VIDEO}")

if __name__ == "__main__":
    create_demo_video()
