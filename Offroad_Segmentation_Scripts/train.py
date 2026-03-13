import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from dataset_loader import OffroadDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing

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
            iou_per_class.append((intersection + smooth) / (union + smooth))
    return np.nanmean(iou_per_class)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    iou_scores = []
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device).long()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        iou = compute_iou(outputs, masks)
        iou_scores.append(iou)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")
        
    return running_loss / len(loader), np.nanmean(iou_scores)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device).long()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            iou = compute_iou(outputs, masks)
            iou_scores.append(iou)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")
            
    return running_loss / len(loader), np.nanmean(iou_scores)

def save_checkpoint(model, optimizer, epoch, iou, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iou': iou,
    }
    torch.save(checkpoint, path)

def main():
    # 1. Configuration
    # Use relative path to dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'Offroad_Segmentation_Training_Dataset')
    
    CHECKPOINT_DIR = './runs/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {DEVICE}")
    
    ENCODER = 'timm-efficientnet-b3'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 10 
    LR = 0.0001
    EPOCHS = 100 
    BATCH_SIZE = 8
    PATIENCE = 10 

    # 2. Initialize Model
    print(f"Initializing DeepLabV3+ with {ENCODER} backbone...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=CLASSES, 
        activation=None,
    ).to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 3. Create Datasets and Loaders
    print("Loading datasets...")
    train_dataset = OffroadDataset(
        os.path.join(DATA_DIR, 'train', 'Color_Images'), 
        os.path.join(DATA_DIR, 'train', 'Segmentation'), 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = OffroadDataset(
        os.path.join(DATA_DIR, 'val', 'Color_Images'), 
        os.path.join(DATA_DIR, 'val', 'Segmentation'), 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 4. Define Loss, Optimizer, and Scheduler
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    criterion = lambda preds, targets: dice_loss(preds, targets) + focal_loss(preds, targets)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # 5. Training Loop
    max_iou = 0
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_iou = validate(model, valid_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > max_iou:
            max_iou = val_iou
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_iou, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"Saved best model with IoU: {max_iou:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_iou, os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth'))

    print("\nTraining complete.")
    print(f"Best Val IoU: {max_iou:.4f}")

if __name__ == "__main__":
    main()
