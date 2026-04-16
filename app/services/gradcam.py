import torch
import cv2
import numpy as np
from app.config.settings import DEVICE
from ml_models.image_model import model

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handle = None

        self._register_hooks()

    def _register_hooks(self):
        self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx, use_smoothing=False, percentile_clip=95):

        output = self.model(input_tensor)

        self.model.zero_grad()

        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()

        cam = cv2.resize(cam, (224, 224))

        threshold = np.percentile(cam, percentile_clip)
        cam = np.minimum(cam, threshold)

        if use_smoothing:
            cam = cv2.bilateralFilter(cam.astype(np.float32), 5, 50, 50)

        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        return cam


gradcam = GradCAM(model, model.layer4[-1])
