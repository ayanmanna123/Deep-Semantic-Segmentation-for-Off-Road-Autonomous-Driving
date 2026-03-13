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

# Custom Dataset for specific file range
class OffroadDatasetRange:
    def __init__(self, images_dir, start_num, end_num, preprocessing=None):
        self.images_dir = images_dir
        self.preprocessing = preprocessing
        self.resize = A.Resize(height=320, width=320)
        
        # Filter files in range
        self.selected_files = []
        for i in range(start_num, end_num + 1):
            filename = f"{i:07d}.png"
            if os.path.exists(os.path.join(images_dir, filename)):
                self.selected_files.append(filename)
        
        print(f"Found {len(self.selected_files)} files in range {start_num}-{end_num}")

    def __getitem__(self, i):
        filename = self.selected_files[i]
        path = os.path.join(self.images_dir, filename)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_image = image.copy()
        
        image_resized = self.resize(image=image)['image']
        
        if self.preprocessing:
            image_pre = self.preprocessing(image=image_resized)['image']
            image_pre = torch.from_numpy(image_pre.transpose(2, 0, 1)).float()
            
        return image_pre, orig_image, filename

    def __len__(self):
        return len(self.selected_files)

def mask_to_rgb(mask):
    COLOR_PALETTE = [
        [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43], [128, 128, 0],
        [255, 105, 180], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
    ]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_PALETTE):
        if i < len(COLOR_PALETTE):
            rgb[mask == i] = color
    return rgb

def main():
    # --- CONFIG ---
    MODEL_PATH = './runs/checkpoints/best_model.pth'
    DATA_DIR = '../Offroad_Segmentation_testImages/Color_Images'
    OUTPUT_DIR = './range_test_results'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    ENCODER = 'resnet50'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    
    # Try finding model in common places
    if not os.path.exists(MODEL_PATH):
        alt_paths = ['./best_model.pth', './runs/best_model.pth', '../best_model.pth']
        for p in alt_paths:
            if os.path.exists(p):
                MODEL_PATH = p
                break
        else:
            print(f"Error: Model not found at {MODEL_PATH}")
            return
            
    print(f"Using model: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    planner = PathPlanner()
    
    dataset = OffroadDatasetRange(DATA_DIR, 81, 91, preprocessing=A.Lambda(image=preprocessing_fn))
    if len(dataset) == 0:
        print("No images found in range.")
        return
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Running range test...")
    
    with torch.no_grad():
        for i, (image, orig_image, filename) in enumerate(tqdm(loader)):
            image = image.to(DEVICE)
            output = model(image)
            
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            path = planner.find_safest_path(mask)
            
            h_orig, w_orig = orig_image.shape[1:3]
            h_model, w_model = mask.shape
            scale_x = w_orig / w_model
            scale_y = h_orig / h_model
            
            scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]
            orig_image_np = orig_image.squeeze().numpy()
            
            path_vis = planner.visualize_on_image(orig_image_np, scaled_path)
            mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            mask_rgb = mask_to_rgb(mask_resized)
            mask_vis = planner.visualize_on_image(mask_rgb, scaled_path, color=(255, 255, 255))
            
            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.imshow(path_vis)
            plt.title(f"Safest Path: {filename[0]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask_vis)
            plt.title("Path on Semantic Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"result_{filename[0]}"))
            plt.close()
            
    print(f"Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
