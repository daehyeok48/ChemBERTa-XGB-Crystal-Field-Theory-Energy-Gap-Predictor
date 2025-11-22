import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path

# Resolve project root and ChemBERTa model path
ROOT = Path(__file__).resolve().parents[1]
CHEMBERTA_PATH = ROOT / "models" / "chemberta"

# Load tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(str(CHEMBERTA_PATH))
model = AutoModelForMaskedLM.from_pretrained(str(CHEMBERTA_PATH))
model.eval()  # Set model to inference mode


@torch.no_grad()
def chemberta_embed(smiles: str) -> np.ndarray:
    """
    Generate a ChemBERTa molecular embedding from a SMILES string.

    The embedding is constructed using:
    - The last 4 hidden layers of the transformer
    - Mean pooling over the token dimension
    - Max pooling over the token dimension
    - Concatenation of (mean + max) pooled vectors

    This produces a contextual molecular representation that captures
    both global and local chemical semantics learned from large SMILES
    corpora.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    np.ndarray
        A 1D NumPy array representing the ChemBERTa embedding
        (dimension = 2 × hidden_size).
    """
    # Tokenize input SMILES
    inputs = tokenizer(
        smiles, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    # Forward pass with hidden states
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple: [layer0, layer1, ... layerN]

    # Average the last 4 layers (context-rich embedding)
    combined = (
        hidden_states[-1][0]
        + hidden_states[-2][0]
        + hidden_states[-3][0]
        + hidden_states[-4][0]
    ) / 4

    # Mean pooling across sequence length
    mean_vec = combined.mean(dim=0)

    # Max pooling across sequence length
    max_vec = combined.max(dim=0).values

    # Final embedding (concatenate mean + max)
    emb = torch.cat([mean_vec, max_vec]).cpu().numpy()

    return emb
