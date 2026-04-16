import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ml_models.LoadDataset import test_loader
from ml_models.image_model import model
from app.config.settings import OUTPUT_DIR, DEVICE
from app.services.gradcam import gradcam

all_preds, all_probs, all_labels, all_paths = [], [], [], []

with torch.no_grad():
    for imgs, labels, paths in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs > 0.5).int()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)


df = pd.DataFrame({
    "image": all_paths,
    "true_label": all_labels,
    "pred_label": all_preds,
    "probability": all_probs
})

df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)


cm = confusion_matrix(all_labels, all_preds)

plt.figure()
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='black', fontsize=12)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()


print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("ROC-AUC:", roc_auc_score(all_labels, all_probs))

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/pr_curve.png")
plt.close()


for i, (imgs, labels, paths) in enumerate(test_loader):
    imgs = imgs.to(DEVICE)
    outputs = model(imgs)

    preds = outputs.argmax(dim=1)

    for j in range(imgs.size(0)):
        img = imgs[j].unsqueeze(0)
        pred_class = preds[j].item()

        cam = gradcam.generate(img, pred_class)

        original = imgs[j].cpu().permute(1, 2, 0).numpy()
        original = (original - original.min()) / (original.max() - original.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = heatmap / 255.0

        overlay = heatmap * 0.4 + original

        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{i}_{j}.png")
        cv2.imwrite(save_path, np.uint8(255 * overlay))