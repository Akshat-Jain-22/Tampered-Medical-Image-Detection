import torch
import timm
import torchvision.models as models
import torch.nn as nn
from app.config.settings import DEVICE, MODEL_PATH
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")
