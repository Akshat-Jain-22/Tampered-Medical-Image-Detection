import joblib
from app.config.settings import METADATA_MODEL
metadata_model = joblib.load(METADATA_MODEL)

def predict_metadata(features):
    return float(metadata_model.predict_proba([features])[0][1])
