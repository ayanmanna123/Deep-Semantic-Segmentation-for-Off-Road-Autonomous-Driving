import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import argparse
from torch.cuda.amp import autocast
from dataset_loader import OffroadDataset, get_validation_augmentation, get_preprocessing

def compute_iou(preds, targets, num_classes=10, smooth=1e-6):
    """Compute IoU for predictions and targets."""
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = (preds == class_id)
        target_inds = (targets == class_id)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append((intersection + smooth) / (union + smooth) if union > 0 else np.nan)
    return iou_per_class

def main():
    # 1. Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='timm-efficientnet-b5', help='Encoder name (e.g. resnet50, timm-efficientnet-b3)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the test dataset')
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation (Horizontal Flip)')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.data_dir:
        DATA_DIR = args.data_dir
    else:
        DATA_DIR = os.path.join(BASE_DIR, 'Offroad_Segmentation_testImages')
        
    if args.model_path:
        MODEL_PATH = args.model_path
    else:
        MODEL_PATH = os.path.join(BASE_DIR, 'Offroad_Segmentation_Scripts', 'runs', 'checkpoints', 'best_model.pth')
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    ENCODER = args.encoder
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 10 

    # 2. Initialize Model
    print(f"Initializing DeepLabV3+ with {ENCODER} backbone...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=CLASSES, 
        activation=None,
    ).to(DEVICE)

    # 3. Load Checkpoint
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Checkpoint not found at {MODEL_PATH}")
        return

    print(f"Loading checkpoint: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')} with IoU: {checkpoint.get('iou', 0.0):.4f}")
    else:
        # Checkpoint is likely a direct state_dict
        model.load_state_dict(checkpoint)
        print(f"Loaded model state_dict directly.")


    # 4. Create Dataset and Loader
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    print(f"Loading test dataset from {DATA_DIR}...")
    test_dataset = OffroadDataset(
        os.path.join(DATA_DIR, 'Color_Images'), 
        os.path.join(DATA_DIR, 'Segmentation'), 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Number of test samples: {len(test_dataset)}")

    # 5. Evaluation
    model.eval()
    all_iou_per_class = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            
            with autocast():
                outputs = model(images)
                
                if args.tta:
                    # Horizontal Flip TTA
                    images_flip = torch.flip(images, dims=[-1])
                    outputs_flip = model(images_flip)
                    outputs_flip = torch.flip(outputs_flip, dims=[-1])
                    
                    # Average probabilities (softmax applied for better averaging)
                    outputs = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_flip, dim=1)) / 2
            
            iou_per_class = compute_iou(outputs, masks, num_classes=CLASSES)
            all_iou_per_class.append(iou_per_class)
            
            current_mean_iou = np.nanmean(np.nanmean(all_iou_per_class, axis=0))
            pbar.set_postfix(mean_iou=f"{current_mean_iou:.4f}")

    # 6. Summary
    all_iou_per_class = np.array(all_iou_per_class)
    mean_iou_per_class = np.nanmean(all_iou_per_class, axis=0)
    overall_mean_iou = np.nanmean(mean_iou_per_class)

    class_names = [
        'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
        'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
    ]

    print("\n" + "="*30)
    print("TEST EVALUATION RESULTS")
    print("="*30)
    for i, iou in enumerate(mean_iou_per_class):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{name:<20}: {iou:.4f}")
    
    print("-" * 30)
    print(f"{'Mean IoU':<20}: {overall_mean_iou:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
