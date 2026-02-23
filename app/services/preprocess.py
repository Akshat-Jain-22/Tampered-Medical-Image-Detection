
import numpy as np
import cv2
import torch
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    RepeatChannel,
    EnsureType,
    ToTensor,
    Compose
)


# ------------------------------------
# Contrast enhancement (CLAHE)
# ------------------------------------
def apply_clahe(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image

# ------------------------------------
# Noise removal
# ------------------------------------
def denoise(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# ------------------------------------
# MONAI base transforms
# ------------------------------------
base_transform = Compose([
    LoadImage(image_only=True),          # PNG, JPG, DICOM
    EnsureChannelFirst(),                # (C, H, W)
    ScaleIntensity(),                    # Normalize intensities
    Resize((380, 380)),                # Model input
    RepeatChannel(repeats=3), 
    ToTensor(),
    EnsureType()                       # Torch tensor
])

# ------------------------------------
# Main preprocessing function
# ------------------------------------
def preprocess_image(image_path: str):
    image = base_transform(image_path)   # MONAI pipeline

    image = image.numpy()

    # Apply enhancement only for grayscale medical images
    if image.shape[0] == 1:
        image = image[0]
        image = apply_clahe(image)
        image = denoise(image)
        image = np.expand_dims(image, axis=0)

    # Convert to 3-channel if needed (for non-medical images)
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Batch dimension

    return torch.tensor(image)
