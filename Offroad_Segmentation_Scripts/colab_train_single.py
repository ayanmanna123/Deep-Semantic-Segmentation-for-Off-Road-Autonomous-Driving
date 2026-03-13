import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ============================================================================
# 1. Dataset Loader Implementation
# ============================================================================

class OffroadDataset(Dataset):
    CLASSES_MAP = {
        100: 0,   # Trees
        200: 1,   # Lush Bushes
        300: 2,   # Dry Grass
        500: 3,   # Dry Bushes
        550: 4,   # Ground Clutter
        600: 5,   # Flowers
        700: 6,   # Logs
        800: 7,   # Rocks
        7100: 8,  # Landscape
        10000: 9  # Sky
    }

    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_val, label_idx in self.CLASSES_MAP.items():
            new_mask[mask == raw_val] = label_idx
        mask = new_mask

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        A.RandomCrop(height=512, width=512),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.OpticalDistortion(distort_limit=0.05, p=0.2),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Resize(height=512, width=512),
    ]
    return A.Compose(test_transform)

def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')
    else:
        return x.astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

# ============================================================================
# 2. Training Loop Utilities
# ============================================================================

def compute_iou(preds, targets, num_classes=10, smooth=1e-6):
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
    running_loss, iou_scores = 0.0, []
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()
        
        with autocast('cuda'):
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
    running_loss, iou_scores = 0.0, []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device).long()
            with autocast('cuda'):
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

# ============================================================================
# 3. Main Training Execution
# ============================================================================

def main():

    # Dataset location
    DATA_DIR = "/content/drive/MyDrive/Offroad_Segmentation_Training_Dataset"

    CHECKPOINT_DIR = "./runs/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {DEVICE}")

    ENCODER = 'timm-efficientnet-b5' # Higher capacity backbone
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 10
    LR = 0.0003 # Adjusted for larger backbone
    EPOCHS = 100
    BATCH_SIZE = 4 # Reduced to avoid OOM with B5 backbone
    PATIENCE = 15  # Increased patience for more complex convergence

    # Initialize Model
    print(f"Initializing DeepLabV3+ with {ENCODER}...")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=CLASSES, activation=None
    ).to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Loaders
    print("Preparing loaders...")
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Class Weights based on observed IoU difficulty
    # Classes: Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky
    weights = torch.tensor([5.0, 25.0, 1.0, 2.0, 25.0, 10.0, 10.0, 15.0, 0.5, 0.1]).to(DEVICE)

    # Combined Loss: Weighted CE + Dice + Focal + Jaccard (Lovász)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    
    criterion = lambda preds, targets: ce_loss(preds, targets) + dice_loss(preds, targets) + focal_loss(preds, targets) + lovasz_loss(preds, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine Annealing with Warm Restarts for better global minima search
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # AMP Scaler
    scaler = GradScaler('cuda')

    # Loop
    max_iou = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        tr_loss, tr_iou = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        va_loss, va_iou = validate(model, valid_loader, criterion, DEVICE)

        scheduler.step()

        print(f"Train Loss: {tr_loss:.4f}, IoU: {tr_iou:.4f} | Val Loss: {va_loss:.4f}, IoU: {va_iou:.4f}")

        if va_iou > max_iou:
            max_iou = va_iou
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, va_iou, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"Best model saved with IoU: {va_iou:.4f}!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, va_iou, os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}.pth'))

    print(f"\nTraining complete. Best Val IoU: {max_iou:.4f}")

if __name__ == "__main__":
    main()
