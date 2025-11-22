import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Morgan Fingerprint (2048 bits)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

# RDKit Descriptor names
desc_names = [d[0] for d in Descriptors.descList]
desc_calc = MolecularDescriptorCalculator(desc_names)


def morgan_fp(smiles: str) -> np.ndarray:
    """
    Morgan Fingerprint (2048-bit)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=float)

    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=float)


def maccs_fp(smiles: str) -> np.ndarray:
    """
    MACCS Keys (167-bit)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=float)

    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=float)


def rdkit_desc(smiles: str) -> np.ndarray:
    """
    RDKit 2D molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(desc_names), dtype=float)

    try:
        values = desc_calc.CalcDescriptors(mol)
        return np.array(values, dtype=float)
    except Exception:
        return np.zeros(len(desc_names), dtype=float)
