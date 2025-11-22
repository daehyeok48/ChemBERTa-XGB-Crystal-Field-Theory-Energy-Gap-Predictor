from rdkit import Chem


def randomize_smiles(smiles: str, n_aug: int = 3):
    """
    Generate randomized SMILES strings for data augmentation.

    RDKit's `MolToSmiles(..., doRandom=True)` creates alternative valid SMILES
    by randomizing the atom traversal order while preserving the chemical
    structure. This increases dataset diversity and improves model robustness.

    Parameters
    ----------
    smiles : str
        Input canonical SMILES string.
    n_aug : int, default=3
        Number of randomized SMILES to generate.

    Returns
    -------
    list of str
        A list containing `n_aug` randomized SMILES strings.
        If the input SMILES is invalid, the original SMILES is returned.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    randomized = []
    for _ in range(n_aug):
        randomized.append(Chem.MolToSmiles(mol, doRandom=True))
    return randomized
