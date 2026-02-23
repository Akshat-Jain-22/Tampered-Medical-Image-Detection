import torch
import timm
import os
from app.config.settings import MODEL_PATH
# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Create model architecture
# -----------------------------
image_model = timm.create_model("efficientnet_b4", pretrained=True)

image_model.classifier = torch.nn.Linear(
    image_model.classifier.in_features,
    2
)

# -----------------------------
# Load trained model
# -----------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ tamper_model.pth not found. Train the model first.")

image_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

image_model.to(device)
image_model.eval()

