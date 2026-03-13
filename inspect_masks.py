import cv2
import numpy as np
import os
import glob

def inspect_masks(mask_dir):
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return

    all_unique_values = set()
    # Check first 10 masks
    for i in range(min(10, len(mask_files))):
        mask = cv2.imread(mask_files[i], cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Failed to load {mask_files[i]}")
            continue
        
        unique_values = np.unique(mask)
        print(f"File: {os.path.basename(mask_files[i])} - Unique values: {unique_values}")
        all_unique_values.update(unique_values)

    print(f"\nAll unique values found in first 10 masks: {sorted(list(all_unique_values))}")

if __name__ == "__main__":
    mask_path = r"c:\Users\manna\Coding\My_github\Hackthon\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    inspect_masks(mask_path)
