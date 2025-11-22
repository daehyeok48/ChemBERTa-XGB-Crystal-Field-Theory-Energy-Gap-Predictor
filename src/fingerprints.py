import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Initialize Morgan Fingerprint generator (2048 bits)
# Captures circular atom environments up to radius=2
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

# Retrieve list of all RDKit 2D descriptors
desc_names = [d[0] for d in Descriptors.descList]
desc_calc = MolecularDescriptorCalculator(desc_names)


def morgan_fp(smiles: str) -> np.ndarray:
    """
    Compute the Morgan fingerprint (ECFP-like) representation.

    Morgan fingerprints encode circular substructures around each atom,
    capturing local chemical environments. They are widely used in
    cheminformatics for similarity search and QSAR/QSPR modeling.

    Parameters
    ----------
    smiles : str
        Input molecular SMILES.

    Returns
    -------
    np.ndarray
        2048-dimensional binary fingerprint vector.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=float)

    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=float)


def maccs_fp(smiles: str) -> np.ndarray:
    """
    Compute MACCS structural keys (167-bit fingerprint).

    MACCS Keys represent the presence or absence of predefined
    chemical substructures (e.g., ring systems, heteroatoms, functional groups).
    This provides an interpretable high-level structural signature.

    Parameters
    ----------
    smiles : str

    Returns
    -------
    np.ndarray
        167-bit structural key vector.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=float)

    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=float)


def rdkit_desc(smiles: str) -> np.ndarray:
    """
    Compute RDKit 2D molecular descriptors.

    Includes broad physicochemical properties such as:
    - LogP, molecular weight
    - Number of H-bond donors/acceptors
    - Topological polar surface area (TPSA)
    - Aromaticity indices
    - Ring counts, charge descriptors, etc.

    These descriptors capture global chemical behavior complementary
    to fingerprints and learned embeddings (ChemBERTa).

    Parameters
    ----------
    smiles : str

    Returns
    -------
    np.ndarray
        Array of RDKit descriptor values (length = len(desc_names)).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(desc_names), dtype=float)

    try:
        values = desc_calc.CalcDescriptors(mol)
        return np.array(values, dtype=float)

    except Exception:
        # Handles occasional calculation failures (e.g., invalid valence)
        return np.zeros(len(desc_names), dtype=float)
