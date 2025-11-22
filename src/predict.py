import numpy as np
import joblib
from pathlib import Path

from .features import build_feature

ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = ROOT / "models" / "scaler.pkl"
MODEL_PATH = ROOT / "models" / "chemberta_trained.pkl"


def predict_single(smiles: str) -> float:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    feat = build_feature(smiles)
    feat_scaled = scaler.transform([feat])

    pred = model.predict(feat_scaled)[0]
    return float(pred)
