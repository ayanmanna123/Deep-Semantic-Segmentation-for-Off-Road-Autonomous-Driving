import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Imports from collinear logic
# (Assuming we have OffroadDataset in the same directory or we redefine it)
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
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids] if masks_dir else None
        self.preprocessing = preprocessing
        self.resize = A.Resize(height=320, width=320)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original for visualization
        orig_image = image.copy()
        
        # Resize image for model
        image = self.resize(image=image)['image']
        
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
            # HWC to CHW
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            
        return image, orig_image, self.ids[i]

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
    # Update these paths to your local data locations
    DATA_DIR = '../Offroad_Segmentation_Training_Dataset/val/Color_Images'
    OUTPUT_DIR = './inference_results'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model (Must match colab_train_single.py)
    ENCODER = 'resnet50'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}. Please download it from Colab.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    
    dataset = OffroadDatasetSimple(DATA_DIR, preprocessing=A.Lambda(image=preprocessing_fn))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Running inference on {len(dataset)} images...")
    
    with torch.no_grad():
        for i, (image, orig_image, filename) in enumerate(tqdm(loader)):
            image = image.to(DEVICE)
            output = model(image)
            
            # Get mask
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize mask back to original size for better visualization
            h, w = orig_image.shape[1:3]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Colorize
            mask_rgb = mask_to_rgb(mask_resized)
            
            # Plot
            orig_image_np = orig_image.squeeze().numpy()
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_image_np)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask_rgb)
            plt.title("Predicted Segmentation")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"pred_{filename[0]}"))
            plt.close()
            
            if i >= 19: # Save first 20 samples
                break

    print(f"Done! Check the '{OUTPUT_DIR}' folder for results.")

if __name__ == "__main__":
    main()
