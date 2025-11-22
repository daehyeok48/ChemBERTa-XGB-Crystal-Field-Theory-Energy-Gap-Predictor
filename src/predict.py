import numpy as np
import joblib
from pathlib import Path

from .features import build_feature

# Resolve project root directory
ROOT = Path(__file__).resolve().parents[1]

# Paths to trained scaler and regression model
SCALER_PATH = ROOT / "models" / "scaler.pkl"
MODEL_PATH = ROOT / "models" / "chemberta_trained.pkl"


def predict_single(smiles: str) -> float:
    """
    Predict the molecular Energy Gap from a SMILES string.

    Workflow:
    1. Load the trained StandardScaler and XGBoost model from disk.
    2. Convert SMILES into the unified feature vector:
       - ChemBERTa embedding (contextual)
       - Morgan fingerprint
       - MACCS keys
       - RDKit 2D descriptors
    3. Apply scaling using the pretrained scaler.
    4. Perform inference using the trained regression model.

    Parameters
    ----------
    smiles : str
        Input SMILES representation of a molecule.

    Returns
    -------
    float
        Predicted Energy Gap value.
    """
    # Load trained components
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    # Build combined feature vector
    feat = build_feature(smiles)

    # Apply scaling
    feat_scaled = scaler.transform([feat])

    # Predict and return scalar value
    pred = model.predict(feat_scaled)[0]
    return float(pred)
