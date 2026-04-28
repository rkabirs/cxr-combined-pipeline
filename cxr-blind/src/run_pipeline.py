"""
run_pipeline.py

Local execution script for running the combined pipeline natively.
"""
import argparse
import os
import pickle
import numpy as np

from src import config
from src.data_loader import load_data, split_patient_data
from src.embeddings import compute_openclip_image_embeddings, compute_hf_vision_embeddings, compute_text_embeddings
from src.retrieval import (
    build_faiss_index, retrieve_top_k,
    build_mof_index, retrieve_top_k_mof,
    label_consistent_blind, mask_to_indices,
)
from src.radgraph_utils import load_radgraph, extract_entities
from src.analysis import build_consensus, connect_blindtype_to_deviation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default=config.BASE_PATH_STR)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    os.environ["RADIOLOGY_BASE_PATH"] = args.base_path

    print(f"Loading data from {args.base_path}...")
    df_clean = load_data(config.REPORTS_FILE, config.PROJECTIONS_FILE)

    clip_cache = config.ARTIFACTS_DIR / "image_embeddings.pkl"
    with open(clip_cache, "rb") as f:
        clip_emb_dict = pickle.load(f)
    df_clean["image_embedding"] = df_clean["filename"].map(clip_emb_dict)
    df_clean = df_clean.dropna(subset=["image_embedding"]).copy()

    dino_cache = config.ARTIFACTS_DIR / "raddino_embeddings.pkl"
    with open(dino_cache, "rb") as f:
        dino_emb_dict = pickle.load(f)
    df_clean["raddino_embedding"] = df_clean["filename"].map(dino_emb_dict)
    df_clean = df_clean.dropna(subset=["raddino_embedding"]).copy()

    train_df, test_df = split_patient_data(df_clean)
    print(f"Train/Test split: Train={len(train_df)}, Test={len(test_df)}")

    train_img_embs  = np.stack(train_df["image_embedding"]).astype("float32")
    test_img_embs   = np.stack(test_df["image_embedding"]).astype("float32")
    train_dino_embs = np.stack(train_df["raddino_embedding"]).astype("float32")
    test_dino_embs  = np.stack(test_df["raddino_embedding"]).astype("float32")

    # --- Retrieval (CLIP-only or MoF) ---
    if config.USE_MOF:
        print("Building MoF (CLIP+DINO) FAISS index...")
        index = build_mof_index(train_img_embs, train_dino_embs)
        D_full, I_full = retrieve_top_k_mof(
            index, test_img_embs, test_dino_embs, config.RETRIEVAL_K + 1
        )
    else:
        print("Building CLIP-only FAISS index...")
        index = build_faiss_index(train_img_embs)
        D_full, I_full = retrieve_top_k(index, test_img_embs, config.RETRIEVAL_K + 1)

    # Strip self-match at rank 0
    D = D_full[:, 1:]
    I = I_full[:, 1:]

    # --- Downstream (identical regardless of retrieval mode) ---
    retrieval_threshold = config.MOF_THRESHOLD if config.USE_MOF else config.CLIP_THRESHOLD
    c_mask, b_mask = label_consistent_blind(
        D, I, test_dino_embs, train_dino_embs,
        retrieval_threshold, config.DINO_THRESHOLD,
    )
    test_cons_neighbors  = mask_to_indices(I, c_mask)
    test_blind_neighbors = mask_to_indices(I, b_mask)

    print(f"Consistent neighbors total: {sum(len(x) for x in test_cons_neighbors)}")
    print(f"Blind neighbors total:      {sum(len(x) for x in test_blind_neighbors)}")


if __name__ == "__main__":
    main()
