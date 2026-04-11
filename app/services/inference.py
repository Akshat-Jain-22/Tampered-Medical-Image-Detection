# Medical Image Tamper Detection Inference
import torch
from ml_models.image_model import model
from app.services.preprocess import transform
from app.services.gradcam import gradcam
import cv2
import numpy as np
import os
from app.config.settings import OUTPUT_DIR, DEVICE

# Initialize GradCAM handler


def run_inference(file_path):
    """
    Run inference on medical image (supports jpg, png, jpeg, dcm)
    Returns: classification, probabilities, and localization heatmap
    """
    try:
        # Load and preprocess image
        img_tensor = transform(file_path).to(DEVICE)

        # Model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)

        pred_class = preds.item()

        # Generate GradCAM heatmap for localization
        cam = gradcam.generate(img_tensor, pred_class)

        # Convert tensor to numpy image
        original = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)

        # Create colored heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = heatmap / 255.0

        # Overlay heatmap on original image
        overlay = heatmap * 0.4 + original
        overlay = np.clip(overlay, 0, 1)

        # Save localization image to output folder
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        gradmap_path = os.path.join(OUTPUT_DIR, f"tamper_localization_{base_filename}.png")
        cv2.imwrite(gradmap_path, np.uint8(255 * overlay[:, :, ::-1]))  # Convert RGB to BGR for cv2.imwrite

        # Prepare response with all required outputs
        response = {
            "classification": "Tampered" if pred_class == 1 else "Authentic",
            "tampered_probability": float(round(probabilities[0][1].item(), 4)),
            "authentic_probability": float(round(probabilities[0][0].item(), 4)),
            "heatmap": overlay.tolist(),  # Convert numpy array to list for JSON serialization
            "heatmap_path": f"/output/{os.path.basename(gradmap_path)}"
        }

        return response

    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")
