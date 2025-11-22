from src.predict import predict_single
from src.visualize import save_molecule_png


def run_cli():
    print("=" * 60)
    print("        ChemBERTa Energy Gap Predictor (CLI Version)")
    print("=" * 60)

    while True:
        smiles = input("\nEnter SMILES ('quit' to exit): ").strip()
        if smiles.lower() == "quit":
            print("\nExiting...")
            break

        try:
            pred = predict_single(smiles)
            print(f"Predicted Energy Gap: {pred:.4f}")

            img_path = save_molecule_png(smiles)
            print(f"Image saved: {img_path}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
