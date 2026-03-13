import os
import sys

# Add Scripts directory to path to import OffroadDataset
scripts_dir = r'C:\Users\manna\Coding\Hackthons\Deep-Semantic-Segmentation-for-Off-Road-Autonomous-Driving\Offroad_Segmentation_Scripts'
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from dataset_loader import OffroadDataset

DATA_DIR = r'C:\Users\manna\Coding\Hackthons\Deep-Semantic-Segmentation-for-Off-Road-Autonomous-Driving\Offroad_Segmentation_Training_Dataset'

def test_loading():
    for set_name in ['train', 'val']:
        images_dir = os.path.join(DATA_DIR, set_name, 'Color_Images')
        masks_dir = os.path.join(DATA_DIR, set_name, 'Segmentation')
        
        print(f"\n--- Checking {set_name} dataset ---")
        print(f"Images dir: {images_dir}")
        if os.path.exists(images_dir):
            files = os.listdir(images_dir)
            print(f"  Files found: {len(files)}")
            non_png = [f for f in files if not f.endswith('.png')]
            if non_png:
                print(f"  Non-png files in images: {non_png}")
        else:
            print(f"  Images dir DOES NOT EXIST")

        print(f"Masks dir: {masks_dir}")
        if os.path.exists(masks_dir):
            files = os.listdir(masks_dir)
            print(f"  Files found: {len(files)}")
            non_png = [f for f in files if not f.endswith('.png')]
            if non_png:
                print(f"  Non-png files in masks: {non_png}")
        else:
            print(f"  Masks dir DOES NOT EXIST")

        try:
            dataset = OffroadDataset(images_dir, masks_dir)
            print(f"{set_name.capitalize()} Dataset initialized with {len(dataset)} items.")
            if len(dataset) > 0:
                img, mask = dataset[0]
                print(f"Successfully loaded first {set_name} item. Image shape: {img.shape}, Mask shape: {mask.shape}")
        except Exception as e:
            print(f"Error during {set_name} dataset loading: {e}")

if __name__ == "__main__":
    test_loading()
