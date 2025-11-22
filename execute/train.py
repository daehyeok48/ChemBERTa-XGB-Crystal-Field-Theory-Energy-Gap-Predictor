import pandas as pd
from pathlib import Path

from src.features import build_dataset
from src.model import train_kfold, train_final_model
from src.utils import banner

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "smiles_dataset.csv"
MODEL_PATH = ROOT / "models" / "chemberta_trained.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


def main():
    banner("TRAINING CHEMBERTA SOTA MODEL")

    df = pd.read_csv(DATA_PATH)
    smiles_list = df["SMILES"].astype(str).tolist()
    y_list = df["Energygap"].values

    # Build dataset with augmentation
    X, Y = build_dataset(smiles_list, y_list, n_aug=3)

    # Evaluate performance via K-Fold
    banner("K-FOLD VALIDATION")
    train_kfold(X, Y, k=5)

    # Train final model + save
    banner("TRAIN FINAL MODEL")
    train_final_model(X, Y, MODEL_PATH, SCALER_PATH)

    banner("TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
