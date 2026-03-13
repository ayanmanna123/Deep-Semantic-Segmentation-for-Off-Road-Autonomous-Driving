import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import random
from path_planner import PathPlanner

class OffroadDatasetSimple:
    CLASSES_MAP = {
        100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
        600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
    }
    
    # Custom Palette to match user's example image
    # Format: [B, G, R] for OpenCV
    COLOR_PALETTE = [
        [0, 150, 0],      # 0: Trees - Dark Green
        [0, 255, 0],      # 1: Lush Bushes - Bright Green
        [0, 255, 255],    # 2: Dry Grass - Yellow
        [0, 100, 0],      # 3: Dry Bushes - Forest Green
        [100, 255, 255],  # 4: Ground Clutter - Light Yellow
        [255, 0, 255],    # 5: Flowers - Magenta
        [0, 0, 255],      # 6: Logs - Red (Obstacle)
        [0, 0, 180],      # 7: Rocks - Dark Red (Obstacle)
        [50, 200, 255],   # 8: Landscape - Golden/Yellowish
        [235, 206, 135],  # 9: Sky - Sky Blue (B, G, R)
    ]

    def __init__(self, images_dir, preprocessing=None):
        all_ids = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter for 90.png to 100.png
        # We look for files that contain '90' through '100' or match a specific numeric range
        self.ids = []
        for fid in all_ids:
            # Try to extract number from filename
            try:
                # Assuming format like '000000090.png' or '90.png'
                num_str = "".join(filter(str.isdigit, fid))
                if num_str:
                    num = int(num_str)
                    if  60<= num <= 1060:
                        self.ids.append(fid)
            except:
                continue
        
        # Sort for consistency
        self.ids.sort()

        # Take 100 random images if there are more than 100
        if len(self.ids) > 100:
            self.ids = random.sample(self.ids, 100)
            self.ids.sort() # Sort again for consistent processing order
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.preprocessing = preprocessing
        self.resize = A.Resize(height=320, width=320)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            # Fallback for failed read
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            return torch.zeros((3, 320, 320)), dummy, self.ids[i]
            
        orig_image = image.copy()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = self.resize(image=image_rgb)['image']
        
        if self.preprocessing:
            image_pre = self.preprocessing(image=image_resized)['image']
            image_pre = torch.from_numpy(image_pre.transpose(2, 0, 1)).float()
        else:
            image_pre = torch.from_numpy(image_resized.transpose(2, 0, 1)).float()
            
        return image_pre, orig_image, self.ids[i]

    def __len__(self):
        return len(self.ids)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(OffroadDatasetSimple.COLOR_PALETTE):
        color_mask[mask == i] = color
    return color_mask

def main():
    # --- CONFIG ---
    # Adjust weights path if needed
    MODEL_PATH = r'C:\Users\manna\Coding\My_Practice\Duality-AI-s-Offroad-Semantic-Scene-Segmentation\Offroad_Segmentation_Scripts\runs\checkpoints\best_model (1).pth'
    # Adjust data dir to point to some images
    DATA_DIR = r'C:\Users\manna\Coding\My_Practice\Duality-AI-s-Offroad-Semantic-Scene-Segmentation\Offroad_Segmentation_Training_Dataset\val\Color_Images'
    OUTPUT_DIR = './final_visualization_results'
    ENCODER = 'timm-efficientnet-b5' 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}. Checking alternative...")
        MODEL_PATH = './runs/checkpoints/epoch_100.pth' # Fallback
        if not os.path.exists(MODEL_PATH):
            print("No model found. Please run training first or provide a 'best_model.pth'.")
            return
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    planner = PathPlanner()
    
    if not os.path.exists(DATA_DIR):
        print(f"Warning: DATA_DIR {DATA_DIR} not found. Visualization will not run.")
        return

    dataset = OffroadDatasetSimple(DATA_DIR, preprocessing=A.Lambda(image=preprocessing_fn))
    if len(dataset) == 0:
        print("No images found in DATA_DIR.")
        return
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Generating side-by-side visualizations for {len(dataset)} images (90.png to 100.png)...")
    
    with torch.no_grad():
        for i, (image_tensor, orig_image, filename_tuple) in enumerate(tqdm(loader)):
            filename = filename_tuple[0]
            image_tensor = image_tensor.to(DEVICE)
            output = model(image_tensor)
            
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Find path on the model-sized mask
            path = planner.find_safest_path(mask)
            
            # Scale path back to original image size
            orig_img_cv = orig_image.squeeze().numpy() # BGR
            h_orig, w_orig = orig_img_cv.shape[:2]
            h_model, w_model = mask.shape
            scale_x = w_orig / w_model
            scale_y = h_orig / h_model
            
            scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]
            
            # Create Overlay
            mask_colored = mask_to_color(mask)
            mask_colored_resized = cv2.resize(mask_colored, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # Blend original image and colored mask (Translucent overlay)
            overlay = cv2.addWeighted(orig_img_cv, 0.5, mask_colored_resized, 0.5, 0)
            
            # Draw Safest Path on Overlay as a thick BLUE ribbon (matching user image)
            if len(scaled_path) > 1:
                pts = np.array(scaled_path, np.int32).reshape((-1, 1, 2))
                
                # Create a blue path effect
                path_mask = np.zeros_like(overlay)
                cv2.polylines(path_mask, [pts], isClosed=False, color=(255, 80, 0), thickness=30) # Solid Blue in BGR
                
                # Blend the path onto the overlay
                path_indices = np.where(np.any(path_mask > 0, axis=-1))
                overlay[path_indices] = cv2.addWeighted(overlay[path_indices], 0.2, path_mask[path_indices], 0.8, 0)
                
                # Add a sharper center line for the path
                cv2.polylines(overlay, [pts], isClosed=False, color=(255, 200, 100), thickness=4)

            # Concatenate side-by-side
            # Add a vertical divider
            divider_width = 10
            divider = np.ones((h_orig, divider_width, 3), dtype=np.uint8) * 200 # Light gray
            combined = np.hstack([orig_img_cv, divider, overlay])
            
            # Save result
            out_path = os.path.join(OUTPUT_DIR, f"final_viz_{filename}")
            cv2.imwrite(out_path, combined)
            
    print(f"Completed! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
