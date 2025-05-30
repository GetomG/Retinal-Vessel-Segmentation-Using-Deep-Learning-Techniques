import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
import tensorflow as tf
from sklearn.utils import shuffle
from albumentations import (
    HorizontalFlip, VerticalFlip,
    ElasticTransform, GridDistortion,
    OpticalDistortion, CoarseDropout
)

H = 512
W = 512

def create_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, augmented=False):

    if augmented:
        train_folder = "train"
        test_folder = "test"
        img_subfolder = "image"
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
    """
    For each (image, mask) pair, optionally apply a suite of augmentations,
    then resize and save out to:
        save_path/image/...
        save_path/mask/...
    """
    # H, W = 512, 512

    # ensure the two target subfolders exist
    create_dir(os.path.join(save_path, "image"))
    create_dir(os.path.join(save_path, "mask"))

    for idx, (img_path, mask_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = os.path.splitext(os.path.basename(img_path))[0]

        # read
        x = cv2.imread(img_path, cv2.IMREAD_COLOR)
        y = imageio.mimread(mask_path)[0]

        # list of (augmented image, mask) pairs
        if augment:
            transforms = [
                HorizontalFlip(p=0.7),
                VerticalFlip(p=0.7),
                ElasticTransform(p=1, alpha=120, sigma=120 * 0.05),
                GridDistortion(p=1),
                OpticalDistortion(p=1, distort_limit=0.05)
            ]

            X, Y = [x], [y]
            for aug in transforms:
                aug_out = aug(image=x, mask=y)
                X.append(aug_out["image"])
                Y.append(aug_out["mask"])
        else:
            X, Y = [x], [y]

        # resize + save each
        for index, (xi, yi) in enumerate(zip(X, Y)):
            xi = clahe_equalized(xi)
            xi = cv2.resize(xi, (W, H))
            yi = cv2.resize(yi, (W, H))

            # choose filename
            suffix = f"_{index}" if len(X) > 1 else ""
            img_name  = f"{name}{suffix}.png"
            mask_name = f"{name}{suffix}.png"

            img_out  = os.path.join(save_path, "image", img_name)
            mask_out = os.path.join(save_path, "mask",  mask_name)

            cv2.imwrite(img_out, xi)
            cv2.imwrite(mask_out, yi)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)              ## (512, 512, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset
