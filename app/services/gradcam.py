import torch
import cv2
from app.config.settings import DEVICE
from ml_models.image_model import model
# GRAD-CAM CLASS for visual explanation
class GradCAM:
    """
    Generates Class Activation Map (Grad-CAM) to visualize which regions
    of the input image are important for the model's prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save feature map activations"""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        """
        Generate Grad-CAM heatmap for specific class

        Args:
            input_tensor: Input image tensor (1, 3, 224, 224)
            class_idx: Class index (0 for Authentic, 1 for Tampered)

        Returns:
            cam: Normalized heatmap (224x224)
        """
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for specific class
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Calculate channel-wise gradient weights
        weights = torch.mean(gradients, dim=(1, 2))

        # Initialize CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        # Weighted combination of activations
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()

        # Resize to input image size
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


gradcam = GradCAM(model, model.layer4[-1])
