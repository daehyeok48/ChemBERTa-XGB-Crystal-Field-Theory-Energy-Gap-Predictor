from src.predict import predict_single
from src.visualize import save_molecule_png


def run_cli():
    """
    Command Line Interface (CLI) for the ChemBERTa Energy Gap Predictor.

    Features:
    - Accepts a SMILES string input from the user.
    - Predicts the molecular energy gap using a pretrained ChemBERTa + XGBoost model.
    - Generates and saves a 2D PNG image of the molecule.
    - Repeats until the user types 'quit'.
    """

    print("=" * 60)
    print("        ChemBERTa Energy Gap Predictor (CLI Version)")
    print("=" * 60)

    while True:
        # Prompt user for SMILES input
        smiles = input("\nEnter SMILES ('quit' to exit): ").strip()
        if smiles.lower() == "quit":
            print("\nExiting...")
            break

        try:
            # Predict energy gap from SMILES
            pred = predict_single(smiles)
            print(f"Predicted Energy Gap: {pred:.4f}")

            # Save 2D molecule PNG and print its path
            img_path = save_molecule_png(smiles)
            print(f"Image saved: {img_path}")

        except Exception as e:
            # Handle errors such as invalid SMILES or model loading issues
            print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
