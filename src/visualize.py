import time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw


def save_molecule_png(smiles: str, save_dir="./molecule_images"):
    """
    Generate and save a 2D PNG image of a molecule from a SMILES string.

    Workflow:
    1. Convert SMILES → RDKit Mol object.
    2. Render a 2D depiction using RDKit's drawing tools.
    3. Save the generated image with a timestamped filename.

    This is used by the CLI predictor to provide a visual representation
    of the input molecule, improving interpretability and user experience.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    save_dir : str, default="./molecule_images"
        Directory where the PNG file will be stored.

    Returns
    -------
    str
        Full file path to the saved PNG image.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Convert SMILES → molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Filename with timestamp for uniqueness
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mol_{ts}.png"
    full_path = save_path / filename

    # RDKit 2D molecule rendering
    img = Draw.MolToImage(mol, size=(400, 400))
    img.save(str(full_path))

    return str(full_path)
