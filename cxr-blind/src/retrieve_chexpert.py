"""
retrieve_chexpert.py

CheXpert label-based FAISS retrieval: each image is represented as a 14-dim
pathology vector (1.0=positive, 0.5=uncertain, 0.0=negative/blank), L2-normalized
and indexed via IndexFlatIP for inner-product nearest-neighbour search.
"""
import numpy as np
import pandas as pd
import faiss
import logging
from . import config

log = logging.getLogger(__name__)

PATHOLOGY_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def _encode(v) -> float:
    """Map a raw CheXpert cell to float: 1.0→1.0, -1.0→0.5, 0.0/NaN→0.0."""
    if pd.isna(v):
        return 0.0
    v = float(v)
    if v == 1.0:
        return 1.0
    if v == -1.0:
        return 0.5
    return 0.0


def load_chexpert_vectors(df: pd.DataFrame) -> np.ndarray:
    """
    Load chexpert_labels.csv and return a float32 array of shape (len(df), 14)
    aligned row-by-row with df.

    Encoding: positive=1.0, uncertain(-1)=0.5, negative/blank=0.0.
    Rows whose uid is absent from the CSV default to all-zeros.
    """
    df_chex = pd.read_csv(config.CHEXPERT_FILE)

    uid_to_vec: dict[str, np.ndarray] = {}
    for _, row in df_chex.iterrows():
        try:
            uid = str(int(float(row["uid"])))
        except (ValueError, TypeError):
            uid = str(row["uid"])
        uid_to_vec[uid] = np.array(
            [_encode(row.get(p)) for p in PATHOLOGY_COLS], dtype=np.float32
        )

    vectors = np.stack([
        uid_to_vec.get(str(row["uid"]), np.zeros(14, dtype=np.float32))
        for _, row in df.iterrows()
    ])
    log.info("Loaded CheXpert vectors: %s", vectors.shape)
    return vectors


def build_chexpert_index(train_vectors: np.ndarray) -> faiss.IndexFlatIP:
    """L2-normalize 14-dim CheXpert vectors and build a FAISS IndexFlatIP."""
    vecs = train_vectors.copy().astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def retrieve_top_k_chexpert(
    index: faiss.IndexFlatIP,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    L2-normalize query vectors and search the CheXpert index.
    Returns (distances, indices), each of shape (M, k).
    """
    qvecs = query_vectors.copy().astype(np.float32)
    faiss.normalize_L2(qvecs)
    return index.search(qvecs, k)
