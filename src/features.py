import numpy as np
from typing import List, Tuple

from .encoder import chemberta_embed
from .fingerprints import morgan_fp, maccs_fp, rdkit_desc
from .augment import randomize_smiles


def build_feature(smiles: str) -> np.ndarray:
    """
    Construct a unified molecular feature vector from a SMILES string.

    The final feature vector concatenates:
    - ChemBERTa embedding (contextual SMILES representation)
    - Morgan fingerprint (circular substructure encoding)
    - MACCS keys (structural motif presence/absence)
    - RDKit 2D descriptors (physicochemical properties)

    This hybrid feature design combines learned sequence embeddings with
    classical cheminformatics descriptors for improved predictive performance.

    Parameters
    ----------
    smiles : str
        Input SMILES representation of a molecule.

    Returns
    -------
    np.ndarray
        1D feature vector combining all descriptor modalities.
    """
    emb = chemberta_embed(smiles)
    fp = morgan_fp(smiles)
    maccs = maccs_fp(smiles)
    desc = rdkit_desc(smiles)
    return np.concatenate([emb, fp, maccs, desc])


def build_dataset(
    smiles_list: List[str], y_list, n_aug: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a full ML-ready dataset with SMILES augmentation.

    For each SMILES entry:
    - Generate `n_aug` randomized SMILES (data augmentation)
    - Extract features for each augmented version
    - Replicate the label accordingly

    This increases dataset diversity, improves model robustness,
    and reduces overfitting for small/medium-sized chemical datasets.

    Parameters
    ----------
    smiles_list : list of str
        List of molecular SMILES.
    y_list : array-like
        Corresponding regression targets (e.g., EnergyGap values).
    n_aug : int, default=3
        Number of randomized SMILES to generate per molecule.

    Returns
    -------
    X : np.ndarray
        2D array of shape (N_samples_augmented, feature_dim)
    Y : np.ndarray
        1D array of shape (N_samples_augmented,)
    """
    X, Y = [], []
    total = len(smiles_list)

    print("[INFO] Building augmented dataset...")

    for i, (sm, y) in enumerate(zip(smiles_list, y_list)):
        # canonical + augmented SMILES
        all_smiles = [sm] + randomize_smiles(sm, n_aug=n_aug)

        for sm_aug in all_smiles:
            X.append(build_feature(sm_aug))
            Y.append(y)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{total}")

    return np.array(X), np.array(Y)
