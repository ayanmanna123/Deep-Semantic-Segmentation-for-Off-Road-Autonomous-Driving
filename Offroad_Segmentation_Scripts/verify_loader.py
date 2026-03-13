import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_loader import OffroadDataset, get_training_augmentation

def visualize_batch(dataset, num_samples=5, out_dir='verify_loader_output'):
    os.makedirs(out_dir, exist_ok=True)
    
    # We use a color palette for visualization [0-9]
    palette = [
        [0, 128, 0],    # Trees - Green
        [0, 255, 0],    # Lush Bushes - Bright Green
        [189, 183, 107],# Dry Grass - DarkKhaki
        [139, 69, 19],  # Dry Bushes - SaddleBrown
        [128, 128, 128],# Ground Clutter - Gray
        [255, 20, 147], # Flowers - DeepPink
        [160, 82, 45],  # Logs - Sienna
        [105, 105, 105],# Rocks - DimGray
        [135, 206, 235],# Landscape - SkyBlue
        [0, 0, 255],    # Sky - Blue
    ]

    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Denormalize is not needed here as we didn't apply image normalization yet in dataset[i]
        # (It's applied in preprocessing, which we haven't passed yet)
        
        # Create a colored mask for visualization
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label_idx, color in enumerate(palette):
            color_mask[mask == label_idx] = color
            
        # Blend image and mask
        # Note: image is already RGB from dataset
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        
        blended = cv2.addWeighted(image_bgr, 0.7, color_mask_bgr, 0.3, 0)
        
        concat = np.hstack([image_bgr, color_mask_bgr, blended])
        
        out_path = os.path.join(out_dir, f"sample_{i}.png")
        cv2.imwrite(out_path, concat)
        print(f"Saved sample to {out_path}")

if __name__ == "__main__":
    train_images = r"c:\Users\manna\Coding\My_github\Hackthon\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    train_masks = r"c:\Users\manna\Coding\My_github\Hackthon\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    
    dataset = OffroadDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        augmentation=get_training_augmentation()
    )
    
    print(f"Dataset size: {len(dataset)}")
    visualize_batch(dataset)
