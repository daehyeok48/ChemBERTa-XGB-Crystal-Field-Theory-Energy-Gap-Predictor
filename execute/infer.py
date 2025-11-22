from src.predict import predict_single
from src.visualize import save_molecule_png
from src.utils import banner


def main():
    """
    Single-molecule prediction script.

    This CLI tools performs:
    - SMILES input from the user
    - ChemBERTa + XGBoost inference to predict the energy gap
    - 2D molecule visualization generated as a PNG image
    - Prints both the numerical prediction and image file path

    Designed for quick testing and demonstration of the trained model.
    """

    # Display module banner
    banner("CHEMBERTA SINGLE PREDICTION")

    # Request SMILES input
    smiles = input("Enter SMILES: ").strip()

    # Predict energy gap using pretrained model
    pred = predict_single(smiles)
    print(f"\nPredicted Energy Gap: {pred:.4f}")

    # Save RDKit molecule PNG with timestamp
    img = save_molecule_png(smiles)
    print(f"Molecule image saved at: {img}")


if __name__ == "__main__":
    # Run CLI interface
    main()
