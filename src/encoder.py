import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHEMBERTA_PATH = ROOT / "models" / "chemberta"

tokenizer = AutoTokenizer.from_pretrained(str(CHEMBERTA_PATH))
model = AutoModelForMaskedLM.from_pretrained(str(CHEMBERTA_PATH))
model.eval()


@torch.no_grad()
def chemberta_embed(smiles: str) -> np.ndarray:
    """
    Compute ChemBERTa embedding from SMILES by:
    - Last 4 Transformer hidden layers
    - Sequence mean pooling
    - Sequence max pooling
    """
    inputs = tokenizer(
        smiles, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of all layers

    # Last 4 layers averaged
    combined = (
        hidden_states[-1][0]
        + hidden_states[-2][0]
        + hidden_states[-3][0]
        + hidden_states[-4][0]
    ) / 4  # shape: (seq_len, hidden_dim)

    # Mean pooling
    mean_vec = combined.mean(dim=0)

    # Max pooling
    max_vec = combined.max(dim=0).values

    # Concatenate (hidden_dim * 2)
    emb = torch.cat([mean_vec, max_vec]).cpu().numpy()

    return emb
