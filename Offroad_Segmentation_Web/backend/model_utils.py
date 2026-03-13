import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from path_planner import PathPlanner

class OffroadDatasetSimple:
    CLASSES_MAP = {
        100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
        600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
    }
    CLASSES = [
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
    for i, color in enumerate(OffroadDatasetSimple.COLOR_PALETTE):
        rgb[mask == i] = color
    return rgb

def get_overlay(image, mask_rgb, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

def get_class_stats(mask):
    unique, counts = np.unique(mask, return_counts=True)
    stats = {}
    total_pixels = mask.size
    
    # Initialize all classes with 0
    for cls in OffroadDatasetSimple.CLASSES:
        stats[cls] = {"count": 0, "percentage": 0.0}
        
    for val, count in zip(unique, counts):
        if 0 <= val < len(OffroadDatasetSimple.CLASSES):
            cls = OffroadDatasetSimple.CLASSES[int(val)]
            stats[cls] = {
                "count": int(count),
                "percentage": float((count / total_pixels) * 100)
            }
    return stats

def process_gt_mask(mask_bytes):
    nparr = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Failed to decode ground truth mask. Please ensure it is a valid image file.")
    
    # Map raw values to 0-9
    h, w = mask.shape[:2]
    mapped_mask = np.zeros((h, w), dtype=np.uint8)
    for raw_val, mapped_val in OffroadDatasetSimple.CLASSES_MAP.items():
        mapped_mask[mask == raw_val] = mapped_val
        
    return mapped_mask

def load_model(model_path, device):
    ENCODER = 'timm-efficientnet-b3'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=None, classes=10, activation=None
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    return model, preprocessing_fn

def predict_on_image(model, preprocessing_fn, image_bytes, device):
    if model is None:
        raise ValueError("Model not loaded. Please check the backend logs for initialization errors.")
        
    # Convert bytes to cv2 image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Please ensure it is a valid image file.")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    # Colorize
    mask_rgb = mask_to_rgb(mask_resized)
    
    return image, mask_rgb, mask_resized

def get_path_visualization(image, mask_indices, mask_rgb):
    """
    Finds the safest path on the mask and draws it on top of a segmentation overlay.
    """
    planner = PathPlanner()
    path = planner.find_safest_path(mask_indices)
    
    h_orig, w_orig = image.shape[:2]
    h_model, w_model = mask_indices.shape
    scale_x = w_orig / w_model
    scale_y = h_orig / h_model
    
    scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]
    
    # Create overlay first (Original Image + Segmented Mask)
    vis_image = get_overlay(image, mask_rgb, alpha=0.5)
    
    if len(scaled_path) > 1:
        pts = np.array(scaled_path, np.int32).reshape((-1, 1, 2))
        
        # Create a blue path effect
        path_mask = np.zeros_like(vis_image)
        # Using RGB: (0, 80, 255) for blue ribbon
        cv2.polylines(path_mask, [pts], isClosed=False, color=(0, 80, 255), thickness=30) 
        
        # Blend the path onto the overlay
        path_indices = np.where(np.any(path_mask > 0, axis=-1))
        vis_image[path_indices] = cv2.addWeighted(vis_image[path_indices], 0.2, path_mask[path_indices], 0.8, 0)
        
        # Add a sharper center line (Light Blue)
        cv2.polylines(vis_image, [pts], isClosed=False, color=(100, 200, 255), thickness=4)

    return vis_image
