from src.predict import predict_single
from src.visualize import save_molecule_png
from src.utils import banner


def main():
    banner("CHEMBERTA SINGLE PREDICTION")

    smiles = input("Enter SMILES: ").strip()

    pred = predict_single(smiles)
    print(f"\nPredicted Energy Gap: {pred:.4f}")

    img = save_molecule_png(smiles)
    print(f"Molecule image saved at: {img}")


if __name__ == "__main__":
    main()
