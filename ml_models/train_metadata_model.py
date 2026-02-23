import numpy as np
import lightgbm as lgb
import joblib
from app.config.settings import METADATA_MODEL
X = np.load("metadata_features.npy")
Y = np.load("labels.npy")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64
)


model.fit(X, Y)
joblib.dump(model, METADATA_MODEL)

print("Model trained & saved.")
