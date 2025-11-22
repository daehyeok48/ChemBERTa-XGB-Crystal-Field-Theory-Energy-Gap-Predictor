import time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw


def save_molecule_png(smiles: str, save_dir="./molecule_images"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mol_{ts}.png"
    full_path = save_path / filename

    img = Draw.MolToImage(mol, size=(400, 400))
    img.save(str(full_path))

    return str(full_path)
