import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage

H = 512
W = 512

def create_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, augmented=False):
    if augmented:
        train_folder = "training"
        test_folder = "test"
        img_subfolder = "images"
        mask_subfolder = "mask"
        img_ext = "*.png"
        mask_ext = "*.png"
    else:
        train_folder = "training"
        test_folder = "test"
        img_subfolder = "images"
        mask_subfolder = "mask"
        img_ext = "*.tif"
        mask_ext = "*.gif"

    train_x = sorted(glob(os.path.join(path, train_folder, img_subfolder, img_ext)))
    train_y = sorted(glob(os.path.join(path, train_folder, mask_subfolder, mask_ext)))

    test_x = sorted(glob(os.path.join(path, test_folder, img_subfolder, img_ext)))
    test_y = sorted(glob(os.path.join(path, test_folder, mask_subfolder, mask_ext)))

    return (train_x, train_y), (test_x, test_y)

def clahe_equalized(img):
    assert len(img.shape) == 3, "Input must be a 3D array (height, width, channels)"
    assert img.shape[2] == 3, "Input must have 3 channels (RGB)"
    assert img.dtype == np.uint8, "Input must be uint8"
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = np.empty(img.shape, dtype=np.uint8)
    for c in range(3):
        img_equalized[:,:,c] = clahe.apply(img[:,:,c])
    return img_equalized

def augment_data(images, masks, save_path, augment=True):
    """Augmentation function remains the same as it uses OpenCV"""
    create_dir(os.path.join(save_path, "training"))
    create_dir(os.path.join(save_path, "test"))
    
    create_dir(os.path.join(save_path, "training", "images"))
    create_dir(os.path.join(save_path, "training", "mask"))
    
    create_dir(os.path.join(save_path, "test", "images"))
    create_dir(os.path.join(save_path, "test", "mask"))

    for idx, (img_path, mask_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = os.path.splitext(os.path.basename(img_path))[0]

        x = cv2.imread(img_path, cv2.IMREAD_COLOR)
        y = imageio.mimread(mask_path)[0]

        if augment:
            transforms = [
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05),
                A.GridDistortion(p=1),
                A.OpticalDistortion(p=1, distort_limit=0.05),
                A.GaussianBlur(p=0.1),
                # A.Normalize(),
                # ToTensorV2(),
            ]

            X, Y = [x], [y]
            for aug in transforms:
                aug_out = aug(image=x, mask=y)
                X.append(aug_out["image"])
                Y.append(aug_out["mask"])
        else:
            X, Y = [x], [y]

        for index, (xi, yi) in enumerate(zip(X, Y)):
            xi = clahe_equalized(xi)
            xi = cv2.resize(xi, (W, H))
            yi = cv2.resize(yi, (W, H))

            suffix = f"_{index}" if len(X) > 1 else ""
            img_name  = f"{name}{suffix}.png"
            mask_name = f"{name}{suffix}.png"

            if 'training' in name:
                img_out  = os.path.join(save_path, "training", "images", img_name)
                mask_out = os.path.join(save_path, "training", "mask",  mask_name)
            elif 'test' in name:
                img_out  = os.path.join(save_path, "test", "images", img_name)
                mask_out = os.path.join(save_path, "test", "mask",  mask_name)
                
            cv2.imwrite(img_out, xi)
            cv2.imwrite(mask_out, yi)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        image = image/255.0
        image = image.astype(np.float32)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        mask = mask/255.0
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # (1, H, W) for PyTorch

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to PyTorch tensors and permute dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask)  # (1, H, W)

        return image, mask

def create_dataloader(X, Y, batch_size, shuffle_data=True, num_workers=0, transform=None):
    dataset = SegmentationDataset(X, Y, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader