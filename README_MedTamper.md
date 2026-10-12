# MedTamper — Medical Image Tampering Detection

**MedTamper** is a deep-learning project for classifying medical images as **Authentic** or **Tampered**. It uses a ResNet-50 image classifier served through a FastAPI web application and can generate a Grad-CAM heatmap for images classified as tampered to help visualize image regions that influenced the model's prediction.

> **Important:** This is an experimental research/educational project, not a clinical diagnostic tool. Predictions and heatmaps can be wrong and must not be used to make medical decisions.

## Features

- **Binary image classification:** Predicts `Authentic` or `Tampered`.
- **Prediction probabilities:** Returns scores for both classes.
- **Grad-CAM visualization:** Generates a heatmap overlay for images predicted as tampered.
- **Medical image preprocessing:** Supports DICOM windowing and standard image preprocessing.
- **Web interface:** Upload an image through the included HTML/CSS/JavaScript frontend.
- **REST API:** FastAPI endpoints for inference and health checks.
- **Training pipeline:** ResNet-50 training with class weighting, weighted sampling, data augmentation, and train/validation loss and accuracy curves.

## Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch, Torchvision
- **Model:** ResNet-50
- **Explainability:** Grad-CAM
- **API / web server:** FastAPI, Uvicorn
- **Image processing:** OpenCV, Albumentations
- **Medical imaging:** pydicom
- **Frontend:** HTML, CSS, JavaScript, Jinja2 templates
- **Evaluation utilities:** scikit-learn, Matplotlib

## Repository Structure

```text
Tampered-Medical-Image-Detection/
├── app/
│   ├── api/routes/          # Detection and health endpoints
│   ├── config/              # Runtime settings and paths
│   ├── services/            # Preprocessing, inference, Grad-CAM
│   ├── static/              # CSS and JavaScript
│   ├── templates/           # HTML pages
│   ├── utils/               # Upload and file validation
│   └── main.py              # FastAPI application
├── ml_models/
│   ├── LoadDataset.py       # Dataset loading and splits
│   ├── image_model.py       # ResNet-50 model loading
│   ├── train_image_model.py # Training script
│   └── test_image_model.py  # Model test script
├── accuracy_curve (1).png   # Training/evaluation plot
├── confusion_matrix (2).png # Confusion matrix plot
├── loss_curve.png           # Loss plot
├── pr_curve.png             # Precision-recall plot
├── roc_curve.png            # ROC plot
└── requirements.txt
```

## How It Works

1. The user uploads a PNG, JPG/JPEG, or DICOM (`.dcm`) file.
2. The application loads the image and resizes/normalizes it for the model.
3. A trained ResNet-50 predicts one of two classes: authentic or tampered.
4. The API returns the predicted class and both class probabilities.
5. If the prediction is tampered, Grad-CAM produces a heatmap overlay to visualize regions that contributed to the prediction.

Grad-CAM is an interpretability aid; it does **not** prove that a region is forged or explain the model's decision with clinical certainty.

## Setup and Installation

### Requirements

- Python 3.10+ recommended (use a version compatible with the pinned packages)
- pip
- A trained model checkpoint named `best_model.pth`
- Dataset files if you want to retrain the model

### 1. Clone the repository

```bash
git clone https://github.com/Akshat-Jain-22/Tampered-Medical-Image-Detection.git
cd Tampered-Medical-Image-Detection
```

If you have renamed the repository, use your new repository URL.

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can depend on your operating system and whether you have a compatible NVIDIA GPU. If the pinned PyTorch packages are unavailable for your environment, install a compatible build from the official PyTorch installation selector and adjust the requirements as needed.

### 4. Add the trained model checkpoint

The inference code expects this file:

```text
ml_models/best_model.pth
```

The checkpoint is not included in this source archive. Obtain the trained checkpoint from your own training run or model artifact storage before starting inference. Do not download or load an untrusted checkpoint.

### 5. Start the application

From the repository root:

```bash
uvicorn app.main:app --reload
```

Open the local URL printed by Uvicorn, usually:

- Home page: `http://127.0.0.1:8000/`
- Image detection page: `http://127.0.0.1:8000/api/detect-medical-tamper`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The model checkpoint must be present for prediction requests to work.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Render the home page |
| `GET` | `/api/detect-medical-tamper` | Render the image detection page |
| `POST` | `/api/detect-medical-tamper` | Upload an image and run inference |
| `GET` | `/health` | Application health check |
| `GET` | `/api/health` | API router health check |

### Example API request

Use `curl` to upload an image:

```bash
curl -X POST \
  -F "file=@path/to/medical_image.png" \
  http://127.0.0.1:8000/api/detect-medical-tamper
```

A successful inference response has this general shape:

```json
{
  "classification": "Tampered",
  "tampered_probability": 0.91,
  "authentic_probability": 0.09,
  "heatmap": null,
  "heatmap_path": "/output/tamper_localization_example_tampered.png"
}
```

The values above are **illustrative only**, not actual model results. The heatmap fields depend on the prediction.

### Supported upload formats

- `.png`
- `.jpg`
- `.jpeg`
- `.dcm` (DICOM)

The application enforces a 50 MB upload limit in its file handler.

## Training the Model

The training code expects the dataset under `DataSet/` in the repository root. Its current folder-to-label mapping is defined in `app/config/settings.py`:

```text
DataSet/
├── TB/  # label 0: authentic
├── TM/  # label 0: authentic
├── FM/  # label 1: tampered
└── FB/  # label 1: tampered
```

Confirm that these folders match the labels and dataset definitions you intend to use before training.

Run training from the repository root:

```bash
python -m ml_models.train_image_model
```

The training script uses a ResNet-50 initialized with ImageNet weights, replaces the final layer with two output classes, and trains for 15 epochs with Adam and a learning rate of `1e-4`. It uses class-weighted cross-entropy and a weighted sampler to address class imbalance. The best checkpoint by validation loss is written to `output/best_model.pth`.

After training, copy the checkpoint to `ml_models/best_model.pth` (the location expected by inference), or update `MODEL_PATH` in `app/config/settings.py` to point to the generated checkpoint.

The dataset is not included in the repository archive. Make sure you have permission to use and redistribute any medical-image data, and remove identifying patient information before sharing data.

## Evaluation Artifacts

The repository contains plots such as:

- Accuracy curve
- Loss curve
- Confusion matrix
- Precision-recall curve
- ROC curve

These plots are provided as project artifacts. Report quantitative performance (accuracy, precision, recall, F1-score, ROC-AUC, dataset size, and split methodology) only after verifying the corresponding evaluation outputs and test procedure. Do not infer a numerical score from the existence of a plot alone.

## Limitations and Responsible Use

- This model is a binary classifier and may not generalize to different modalities, scanners, acquisition settings, or tampering techniques.
- Performance depends on dataset quality, label quality, class balance, and data leakage prevention.
- Grad-CAM visualizations are approximate explanations, not tamper masks or proof of manipulation.
- The current app configuration uses permissive CORS (`allow_origins=["*"]`). Restrict allowed origins before deployment.
- Do not deploy with real patient data until access controls, privacy, logging, retention, and security requirements have been reviewed.
- **Not for clinical use:** A qualified medical professional must independently review all relevant evidence.

## Future Improvements

- Add automated unit and API tests.
- Add calibrated thresholds and richer evaluation reporting.
- Evaluate generalization on an independent, patient-level held-out dataset.
- Add safer upload handling, rate limiting, and production CORS configuration.
- Package model artifacts with checksums and reproducible versioning.
- Improve explanation overlays and validate them against annotated tamper regions.

## Contributing

Issues and pull requests are welcome. For substantial changes, open an issue first to discuss the proposed improvement.

## License

No license is currently specified. Add a `LICENSE` file before distributing the project for reuse.
