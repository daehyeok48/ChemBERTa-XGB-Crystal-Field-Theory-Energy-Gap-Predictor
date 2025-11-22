from rdkit import Chem


def randomize_smiles(smiles: str, n_aug: int = 3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    randomized = []
    for _ in range(n_aug):
        randomized.append(Chem.MolToSmiles(mol, doRandom=True))
    return randomized
