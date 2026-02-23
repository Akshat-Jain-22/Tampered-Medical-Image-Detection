# Explain this code line by line and Optimize it


# import torch
# import cv2
# import numpy as np

# def generate_gradcam(model, image_tensor):
#     gradients = []
#     activations = []

#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     target_layer = model.layer4[1].conv2
#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_backward_hook(backward_hook)

#     output = model(image_tensor)
#     class_idx = output.argmax()
#     output[0, class_idx].backward()

#     grads = gradients[0].mean(dim=[2,3], keepdim=True)
#     cam = (grads * activations[0]).sum(dim=1).squeeze()
#     cam = cam.detach().cpu().numpy()

#     cam = np.maximum(cam, 0)
#     cam = cam / cam.max()
#     return cam



# import cv2
# import numpy as np
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from ml_models.image_model import model

# target_layer = model.conv_head   # Best layer in EfficientNet for CAM

# cam = GradCAM(model=model, target_layers=[target_layer])

# def generate_cam(image_tensor, original_image):
#     targets = [ClassifierOutputTarget(1)]  # class = Tampered

#     grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

#     original_image = original_image / 255.0

#     heatmap = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
#     return heatmap
