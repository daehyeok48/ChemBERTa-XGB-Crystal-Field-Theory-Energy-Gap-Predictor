import numpy as np
from typing import List, Tuple

from .encoder import chemberta_embed
from .fingerprints import morgan_fp, maccs_fp, rdkit_desc
from .augment import randomize_smiles


def build_feature(smiles: str) -> np.ndarray:
    emb = chemberta_embed(smiles)
    fp = morgan_fp(smiles)
    maccs = maccs_fp(smiles)
    desc = rdkit_desc(smiles)
    return np.concatenate([emb, fp, maccs, desc])


def build_dataset(
    smiles_list: List[str], y_list, n_aug: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    total = len(smiles_list)

    print("[INFO] Building augmented dataset...")

    for i, (sm, y) in enumerate(zip(smiles_list, y_list)):
        all_smiles = [sm] + randomize_smiles(sm, n_aug=n_aug)

        for sm_aug in all_smiles:
            X.append(build_feature(sm_aug))
            Y.append(y)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{total}")

    return np.array(X), np.array(Y)
