# Image Preprocessing and Transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pydicom

# Augmentation for training
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

# Transformations for validation/inference
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

def load_image(file_path):
    """
    Load standard image (jpg, png, jpeg)

    Args:
        file_path: Path to image file

    Returns:
        np.ndarray: Image array in RGB format
    """
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_dicom(file_path):
    """
    Load DICOM medical image and convert to RGB

    Args:
        file_path: Path to DICOM file

    Returns:
        np.ndarray: Image array in RGB format
    """
    try:
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array

        # Normalize pixel values to 0-255 range
        if img.dtype != np.uint8:
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max - img_min > 0:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    except Exception as e:
        raise ValueError(f"Failed to load DICOM file: {str(e)}")

def transform(file_path):
    """
    Load medical image and apply validation transforms

    Supports: jpg, png, jpeg, dcm files

    Args:
        file_path: Path to image file

    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, 224, 224)
    """
    # Load appropriate format
    if file_path.lower().endswith('.dcm'):
        img = load_dicom(file_path)
    else:
        img = load_image(file_path)

    # Apply transformations and return as batch tensor
    transformed = val_transform(image=img)
    return transformed['image'].unsqueeze(0)

