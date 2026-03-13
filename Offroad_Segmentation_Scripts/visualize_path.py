import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from path_planner import PathPlanner

# Imports from collinear logic
class OffroadDatasetSimple:
    CLASSES_MAP = {
        100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
        600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
    }
    COLOR_PALETTE = [
        [34, 139, 34],    # Trees - forest green
        [0, 255, 0],      # Lush Bushes - lime
        [210, 180, 140],  # Dry Grass - tan
        [139, 90, 43],    # Dry Bushes - brown
        [128, 128, 0],    # Ground Clutter - olive
        [255, 105, 180],  # Flowers - pink
        [139, 69, 19],    # Logs - saddle brown
        [128, 128, 128],  # Rocks - gray
        [160, 82, 45],    # Landscape - sienna
        [135, 206, 235],  # Sky - sky blue
    ]

    def __init__(self, images_dir, masks_dir=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.preprocessing = preprocessing
        self.resize = A.Resize(height=320, width=320)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_image = image.copy()
        
        image_resized = self.resize(image=image)['image']
        
        if self.preprocessing:
            image_pre = self.preprocessing(image=image_resized)['image']
            image_pre = torch.from_numpy(image_pre.transpose(2, 0, 1)).float()
            
        return image_pre, orig_image, self.ids[i]

    def __len__(self):
        return len(self.ids)

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(OffroadDatasetSimple.COLOR_PALETTE):
        rgb[mask == i] = color
    return rgb

def main():
    # --- CONFIG ---
    MODEL_PATH = './runs/checkpoints/best_model.pth'
    DATA_DIR = '../Offroad_Segmentation_Training_Dataset/val/Color_Images'
    OUTPUT_DIR = './path_visualization_results'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    ENCODER = 'resnet50' # or whatever was used in colab_train_single.py
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}. Searching in common locations...")
        # Check alternative locations
        alt_paths = ['./best_model.pth', './runs/best_model.pth', '../best_model.pth']
        for p in alt_paths:
            if os.path.exists(p):
                MODEL_PATH = p
                print(f"Found model at: {MODEL_PATH}")
                break
        else:
            print("Please ensure your trained model is available.")
            return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    planner = PathPlanner()
    
    dataset = OffroadDatasetSimple(DATA_DIR, preprocessing=A.Lambda(image=preprocessing_fn))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Generating safest paths for {min(len(dataset), 10)} images...")
    
    with torch.no_grad():
        for i, (image, orig_image, filename) in enumerate(tqdm(loader)):
            image = image.to(DEVICE)
            output = model(image)
            
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Find path on the model-sized mask
            path = planner.find_safest_path(mask)
            
            # Scale path back to original image size
            h_orig, w_orig = orig_image.shape[1:3]
            h_model, w_model = mask.shape
            scale_x = w_orig / w_model
            scale_y = h_orig / h_model
            
            scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]
            
            # Visualizations
            orig_image_np = orig_image.squeeze().numpy()
            
            # 1. Image with Path
            path_vis = planner.visualize_on_image(orig_image_np, scaled_path)
            
            # 2. Mask with Path
            mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            mask_rgb = mask_to_rgb(mask_resized)
            mask_vis = planner.visualize_on_image(mask_rgb, scaled_path, color=(255, 255, 255))
            
            # Plot results
            plt.figure(figsize=(15, 7))
            
            plt.subplot(1, 2, 1)
            plt.imshow(path_vis)
            plt.title("Safest Path on Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask_vis)
            plt.title("Path on Semantic Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"path_{filename[0]}"))
            plt.close()
            
            if i >= 9: # Do 10 samples
                break
                
    print(f"Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
