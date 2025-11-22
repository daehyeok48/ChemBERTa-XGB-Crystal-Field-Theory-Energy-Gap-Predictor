import pandas as pd
from pathlib import Path

from src.features import build_dataset
from src.model import train_kfold, train_final_model
from src.utils import banner

# Resolve project root directory
ROOT = Path(__file__).resolve().parents[1]

# Paths for dataset & output model files
DATA_PATH = ROOT / "data" / "smiles_dataset.csv"
MODEL_PATH = ROOT / "models" / "chemberta_trained.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


def main():
    """
    Main training pipeline for the ChemBERTa-based molecular property model.

    Workflow:
    1. Load SMILES dataset (SMILES → target property).
    2. Build feature matrix using:
       - ChemBERTa contextual embeddings
       - Morgan fingerprint
       - MACCS keys
       - RDKit 2D descriptors
       - SMILES augmentation
    3. Evaluate performance using 5-fold cross validation.
    4. Train final XGBoost model on full dataset.
    5. Save trained model and fitted scaler for inference.

    This script produces:
    - chemberta_trained.pkl  (XGBoost model)
    - scaler.pkl             (StandardScaler for feature normalization)
    """

    banner("TRAINING CHEMBERTA SOTA MODEL")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    smiles_list = df["SMILES"].astype(str).tolist()
    y_list = df["Energygap"].values

    # Feature extraction with SMILES augmentation
    X, Y = build_dataset(smiles_list, y_list, n_aug=3)

    # Step 1: K-fold validation
    banner("K-FOLD VALIDATION")
    train_kfold(X, Y, k=5)

    # Step 2: Final model fit + save
    banner("TRAIN FINAL MODEL")
    train_final_model(X, Y, MODEL_PATH, SCALER_PATH)

    banner("TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
