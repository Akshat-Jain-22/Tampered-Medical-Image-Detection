import os

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "ml_models/tamper_model.pth")
METADATA_MODEL = os.path.join(BASE_DIR, "ml_models/metadata_model.pkl")
DATASET_PATH_OPEN = os.path.join(BASE_DIR, "DataSet/Experiment_Open")
DATASET_PATH_BLIND = os.path.join(BASE_DIR, "DataSet/Experiment_Blind")

# Label mapping
label_map = {
    "TB": 0,   # Authentic
    "TM": 0,   # Authentic
    "FB": 1,   # Tampered
    "FM": 1    # Tampered
}
