import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
import argparse
from tqdm import tqdm

# Configuration
CLASSES_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

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

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_PALETTE):
        rgb[mask == i] = color
    return rgb

def load_model(model_path, device):
    ENCODER = 'resnet50'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    return model, preprocessing_fn

def predict(model, preprocessing_fn, image, device):
    orig_h, orig_w = image.shape[:2]
    
    # Resize for model
    resize = A.Resize(height=320, width=320)
    resized_image = resize(image=image)['image']
    
    # Preprocess
    preprocessed = A.Lambda(image=preprocessing_fn)(image=resized_image)['image']
    # HWC to CHW
    input_tensor = torch.from_numpy(preprocessed.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results on a single test image.")
    parser.add_argument("--image_path", type=str, default="../Offroad_Segmentation_testImages/Color_Images/0000060.png", help="Path to the test image.")
    parser.add_argument("--mask_path", type=str, default="../Offroad_Segmentation_testImages/Segmentation/0000060.png", help="Path to the ground truth mask (optional).")
    parser.add_argument("--model_path", type=str, default="./runs/checkpoints/best_model.pth", help="Path to the model weights.")
    parser.add_argument("--output_path", type=str, default="full_report.png", help="Path to save the full report image.")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    print("Loading model...")
    model, preprocessing_fn = load_model(args.model_path, DEVICE)

    # 2. Load Image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Predict
    print("Running inference...")
    pred_mask = predict(model, preprocessing_fn, image, DEVICE)
    pred_rgb = mask_to_rgb(pred_mask)

    # 4. Process Ground Truth (if exists)
    gt_mask_mapped = None
    gt_rgb = None
    if os.path.exists(args.mask_path):
        gt_mask_raw = cv2.imread(args.mask_path, cv2.IMREAD_UNCHANGED)
        gt_mask_mapped = np.zeros_like(gt_mask_raw, dtype=np.uint8)
        for val, idx in CLASSES_MAP.items():
            gt_mask_mapped[gt_mask_raw == val] = idx
        gt_rgb = mask_to_rgb(gt_mask_mapped)

    # 5. Calculate class distribution
    unique, counts = np.unique(pred_mask, return_counts=True)
    stats = dict(zip(unique, counts))
    total_pixels = pred_mask.size
    
    print("\nClass Statistics (Prediction):")
    for i in range(10):
        count = stats.get(i, 0)
        percentage = (count / total_pixels) * 100
        print(f"  {CLASS_NAMES[i]:<15}: {count:>10} pixels ({percentage:>6.2f}%)")

    # 6. Visualization
    print("\nGenerating comprehensive report...")
    fig = plt.figure(figsize=(24, 16))
    plt.suptitle(f"Segmentations Inference Report: {os.path.basename(args.image_path)}", fontsize=24)

    # Row 1: Images
    # Original Image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=16)
    ax1.axis('off')

    # Predicted Mask
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(pred_rgb)
    ax2.set_title("Predicted Segmentation", fontsize=16)
    ax2.axis('off')

    # Overlay
    ax3 = fig.add_subplot(2, 3, 3)
    overlay = cv2.addWeighted(image, 0.6, pred_rgb, 0.4, 0)
    ax3.imshow(overlay)
    ax3.set_title("Overlay (Original + Prediction)", fontsize=16)
    ax3.axis('off')

    # Row 2: Metrics and GT
    # Ground Truth Mask
    if gt_rgb is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(gt_rgb)
        ax4.set_title("Ground Truth Mask", fontsize=16)
        ax4.axis('off')
        
        # Calculate IoU for this image if GT exists
        intersection = np.logical_and(pred_mask == gt_mask_mapped, gt_mask_mapped != 255) # ignore background if any
        # (Simplified IoU for visualization)
        # Note: Proper mIoU would iterate per class
    else:
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.text(0.5, 0.5, "Ground Truth\nNot Available", ha='center', va='center', fontsize=20, color='gray')
        ax4.set_title("Ground Truth Mask", fontsize=16)
        ax4.axis('off')

    # Class Distribution Bar Chart
    ax5 = fig.add_subplot(2, 3, 5)
    class_counts = [stats.get(i, 0) for i in range(10)]
    colors = [np.array(c)/255.0 for c in COLOR_PALETTE]
    ax5.barh(CLASS_NAMES, class_counts, color=colors)
    ax5.set_title("Predicted Class Distribution (Pixel Count)", fontsize=16)
    ax5.invert_yaxis()
    ax5.set_xlabel("Count", fontsize=12)

    # Prediction Summary Text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    info_text = f"Report for: {os.path.basename(args.image_path)}\n\n"
    info_text += f"Model Architecture: DeepLabV3+\n"
    info_text += f"Encoder: ResNet50\n"
    info_text += f"Input Size: 320x320 (interpolated back to {image.shape[1]}x{image.shape[0]})\n\n"
    info_text += "Class Distribution (% of image):\n"
    info_text += "-" * 40 + "\n"
    for i in range(10):
        percentage = (stats.get(i, 0) / total_pixels) * 100
        info_text += f"{CLASS_NAMES[i]:<15}: {percentage:>6.2f}%\n"
    
    ax6.text(0.05, 0.1, info_text, transform=ax6.transAxes, fontsize=14, family='monospace', verticalalignment='bottom')
    ax6.set_title("Inference Summary", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(args.output_path, dpi=150)
    print(f"\nReport successfully saved as: {os.path.abspath(args.output_path)}")
    
    # Also show the plot if possible
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
