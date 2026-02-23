from ml_models.train_image_model import model, device
from ml_models.train_image_model import blind_loader
from sklearn.metrics import accuracy_score, recall_score, precision_score, zero_one_loss
import torch
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in blind_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1).cpu()

        y_true.extend(labels)
        y_pred.extend(preds)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Tamper Recall:", recall_score(y_true, y_pred))
print("Tamper Precision:", precision_score(y_true, y_pred))
print("Zero-One Loss:", zero_one_loss(y_true, y_pred))
