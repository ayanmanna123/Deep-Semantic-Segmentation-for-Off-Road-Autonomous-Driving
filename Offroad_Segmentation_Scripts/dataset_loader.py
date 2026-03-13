import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class OffroadDataset(Dataset):
    """
    Custom Dataset for Offroad Semantic Segmentation.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        classes (list): list of class names (not strictly used for mapping here but good for reference)
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, contrast, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    """
    
    # Mapping from raw pixel values to label indices [0, 9]
    VALUE_MAP = {
        0: 0,        # background (if any)
        100: 1,      # Trees
        200: 2,      # Lush Bushes
        300: 3,      # Dry Grass
        500: 4,      # Dry Bushes
        550: 5,      # Ground Clutter
        600: 6,      # Flowers
        700: 7,      # Logs
        800: 8,      # Rocks
        7100: 9,     # Landscape
        10000: 10    # Sky (Total 11 classes if 0 is background, or 10 if we map 10000 to 9)
    }

    # Based on README.md, let's stick to the 10 classes provided in the table
    # and map them to 0-9. If 0 is background, we can map 100->0, 200->1...
    
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

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
        
        # Map raw mask values to class labels [0, 9]
        # We initialize with a value that we can treat as 'ignore' or 'background' if needed
        # But here we map everything based on CLASSES_MAP
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_val, label_idx in self.CLASSES_MAP.items():
            new_mask[mask == raw_val] = label_idx
            
        mask = new_mask
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
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
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height=320, width=320, always_apply=True),
    ]
    return A.Compose(test_transform)

def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')
    else:
        return x.astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)
