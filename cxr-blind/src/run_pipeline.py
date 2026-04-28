"""
run_pipeline.py

Local execution script for the combined CXR blind retrieval pipeline.
Supports three retrieval modes: 'baseline' (CLIP-only), 'mof' (CLIP+DINO), 'chexpert'.
"""
import argparse
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.analysis import build_consensus, connect_blindtype_to_deviation
from src.data_loader import load_data, split_patient_data
from src.radgraph_utils import extract_entities, load_radgraph
from src.retrieve_chexpert import (
    build_chexpert_index,
    load_chexpert_vectors,
    retrieve_top_k_chexpert,
)
from src.retrieval import (
    build_faiss_index,
    build_mof_index,
    label_consistent_blind,
    mask_to_indices,
    retrieve_top_k,
    retrieve_top_k_mof,
)


def _run_visualizations(blind_pairs_df: pd.DataFrame, test_df: pd.DataFrame, results_dir):
    types = ["Type 1", "Type 2", "Type 3"]
    type_counts = blind_pairs_df["blind_type"].value_counts()

    print("\nBlind Retrieval Types Breakdown:")
    print(type_counts)

    print("\nTop Deviant RadGraph Entities per Pathology:")
    for pathology, group in blind_pairs_df.groupby("primary_pathology"):
        missing_list = [e for m in group["missing_entities"] for e in m]
        if missing_list:
            print(f"\n-- {pathology} --")
            for ent, count in Counter(missing_list).most_common(5):
                print(f"  {count}x : {ent}")

    # Bar chart: blind pair types
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = [type_counts.get(t, 0) for t in types]
    bars = ax.bar(types, counts, color=["#4878CF", "#6ACC65", "#D65F5F"], width=0.5)
    for bar, num in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(num),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Blind Retrieval Pairs by Error Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(results_dir / "blind_pair_types.png", dpi=150)
    plt.close(fig)

    # Frequency table: top missing entities by type
    all_missing = [e for m in blind_pairs_df["missing_entities"] for e in m]
    top_entities = [ent for ent, _ in Counter(all_missing).most_common(10)]

    freq_data = []
    for t in types:
        type_df = blind_pairs_df[blind_pairs_df["blind_type"] == t]
        type_total = len(type_df)
        type_missing = [e for m in type_df["missing_entities"] for e in m]
        type_counts_dict = Counter(type_missing)
        for ent in top_entities:
            count = type_counts_dict.get(ent, 0)
            proportion = (count / type_total * 100) if type_total > 0 else 0
            freq_data.append({
                "Entity": ent,
                "Type": t,
                "Count": count,
                "Prevalence (%)": round(proportion, 1),
            })

    freq_df = pd.DataFrame(freq_data)
    freq_df.to_csv(results_dir / "top_missing_entities_by_type.csv", index=False)
    print(f"Table saved to {results_dir / 'top_missing_entities_by_type.csv'}")

    pivot_df = freq_df.pivot(index="Entity", columns="Type", values="Prevalence (%)").fillna(0)

    # Grouped bar plot
    x = np.arange(len(top_entities))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, pivot_df.get("Type 1", 0), width, label="Type 1", color="#4878CF")
    ax.bar(x,         pivot_df.get("Type 2", 0), width, label="Type 2", color="#6ACC65")
    ax.bar(x + width, pivot_df.get("Type 3", 0), width, label="Type 3", color="#D65F5F")
    ax.set_ylabel("Prevalence (% of Pairs Missing Entity)")
    ax.set_title("Top Missing RadGraph Entities by Blind Error Type (Normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(top_entities, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(results_dir / "missing_entities_grouped_bar.png", dpi=150)
    plt.close(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pivot_df.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(types)))
    ax.set_yticks(np.arange(len(top_entities)))
    ax.set_xticklabels(types)
    ax.set_yticklabels(top_entities)
    ax.set_title("Heatmap: Prevalence (%) of Missing Entities per Error Type")
    for i in range(len(top_entities)):
        for j in range(len(types)):
            val = pivot_df.values[i, j]
            color = "black" if val < pivot_df.values.max() / 2 else "white"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color)
    fig.colorbar(cax, label="Prevalence (%)")
    plt.tight_layout()
    fig.savefig(results_dir / "missing_entities_heatmap.png", dpi=150)
    plt.close(fig)

    # Deviation boxplot by error type
    blind_pairs_df = blind_pairs_df.copy()
    blind_pairs_df["query_uid_str"] = blind_pairs_df["query_uid"].astype(str)
    test_df = test_df.copy()
    test_df["uid_str"] = test_df["uid"].astype(str)
    dev_df = pd.merge(
        blind_pairs_df,
        test_df[["uid_str", "radgraph_deviation_full"]],
        left_on="query_uid_str", right_on="uid_str",
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_data, labels = [], []
    for t in types:
        vals = dev_df[dev_df["blind_type"] == t]["radgraph_deviation_full"].dropna().values
        plot_data.append(vals if len(vals) > 0 else [0])
        labels.append(t)
    bp = ax.boxplot(plot_data, patch_artist=True, labels=labels,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], ["#4878CF", "#6ACC65", "#D65F5F"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("RadGraph Deviation from Consensus")
    ax.set_title("Absolute RadGraph Deviation vs. Error Type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(results_dir / "radgraph_deviation_vs_type.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default=config.BASE_PATH_STR)
    parser.add_argument(
        "--mode", type=str, default=config.RETRIEVAL_MODE,
        choices=["baseline", "mof", "chexpert"],
    )
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    os.environ["RADIOLOGY_BASE_PATH"] = args.base_path
    config.RETRIEVAL_MODE = args.mode
    config.RESULTS_DIR = config.BASE_PATH / config._results_dir_map[args.mode]
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== CXR Pipeline | mode={args.mode} | base={args.base_path} ===")

    # --- 1. Data ---
    print("\n--- 1. Loading dataset ---")
    df_clean = load_data(config.REPORTS_FILE, config.PROJECTIONS_FILE)

    # --- 2. CLIP embeddings (from cache) ---
    print("\n--- 2. BioMedCLIP image embeddings ---")
    clip_cache = config.ARTIFACTS_DIR / "image_embeddings.pkl"
    with open(clip_cache, "rb") as f:
        clip_emb_dict = pickle.load(f)
    df_clean["image_embedding"] = df_clean["filename"].map(clip_emb_dict)
    df_clean = df_clean.dropna(subset=["image_embedding"]).copy()

    # --- 3. RAD-DINO embeddings (from cache) ---
    print("\n--- 3. RAD-DINO image embeddings ---")
    dino_cache = config.ARTIFACTS_DIR / "raddino_embeddings.pkl"
    with open(dino_cache, "rb") as f:
        dino_emb_dict = pickle.load(f)
    df_clean["raddino_embedding"] = df_clean["filename"].map(dino_emb_dict)
    df_clean = df_clean.dropna(subset=["raddino_embedding"]).copy()

    # Reset index so positional slicing into entity lists is safe after split
    df_clean = df_clean.reset_index(drop=True)

    train_df, test_df = split_patient_data(df_clean)
    print(f"Train/Test split: Train={len(train_df)}, Test={len(test_df)}")

    train_img_embs  = np.stack(train_df["image_embedding"]).astype("float32")
    test_img_embs   = np.stack(test_df["image_embedding"]).astype("float32")
    train_dino_embs = np.stack(train_df["raddino_embedding"]).astype("float32")
    test_dino_embs  = np.stack(test_df["raddino_embedding"]).astype("float32")

    # --- 4. Retrieval ---
    print(f"\n--- 4. FAISS retrieval ({args.mode.upper()}) ---")
    if args.mode == "mof":
        index = build_mof_index(train_img_embs, train_dino_embs)
        D_full, I_full = retrieve_top_k_mof(
            index, test_img_embs, test_dino_embs, config.RETRIEVAL_K + 1
        )
        retrieval_threshold = config.MOF_THRESHOLD
    elif args.mode == "chexpert":
        print(f"Loading CheXpert vectors from {config.CHEXPERT_FILE}...")
        train_chex = load_chexpert_vectors(train_df)
        test_chex  = load_chexpert_vectors(test_df)
        index = build_chexpert_index(train_chex)
        D_full, I_full = retrieve_top_k_chexpert(index, test_chex, config.RETRIEVAL_K + 1)
        retrieval_threshold = config.CHEXPERT_THRESHOLD
    else:  # baseline
        index = build_faiss_index(train_img_embs)
        D_full, I_full = retrieve_top_k(index, test_img_embs, config.RETRIEVAL_K + 1)
        retrieval_threshold = config.CLIP_THRESHOLD

    # Strip closest neighbor (self-match in full-dataset mode; nearest in split mode)
    D = D_full[:, 1:]
    I = I_full[:, 1:]

    # --- 5. Consistent / blind labeling (RAD-DINO cross-check for all modes) ---
    c_mask, b_mask = label_consistent_blind(
        D, I, test_dino_embs, train_dino_embs,
        retrieval_threshold, config.DINO_THRESHOLD,
    )
    test_cons_neighbors  = mask_to_indices(I, c_mask)
    test_blind_neighbors = mask_to_indices(I, b_mask)

    print(f"Consistent neighbors total: {sum(len(x) for x in test_cons_neighbors)}")
    print(f"Blind neighbors total:      {sum(len(x) for x in test_blind_neighbors)}")

    # --- 6. RadGraph entity extraction ---
    print("\n--- 5. RadGraph entity extraction ---")
    rg_cache = config.ARTIFACTS_DIR / "full_dataset_radgraph_entities.pkl"
    if rg_cache.exists() and not args.skip_cache:
        print(f"Loading RadGraph cache from {rg_cache}")
        with open(rg_cache, "rb") as f:
            full_ents = pickle.load(f)
    else:
        rg_model = load_radgraph(config.RADGRAPH_MODEL)
        full_ents = extract_entities(
            df_clean["full_text"].tolist(), rg_model, config.RADGRAPH_BATCH_SIZE
        )
        with open(rg_cache, "wb") as f:
            pickle.dump(full_ents, f)

    # Slice entity lists to align with train/test split (df_clean has contiguous index)
    train_ents = [full_ents[i] for i in train_df.index]
    test_ents  = [full_ents[i] for i in test_df.index]

    # --- 7. Consensus & CheXpert analysis ---
    print("\n--- 6. Consensus & CheXpert analysis ---")
    test_df = build_consensus(
        test_df, train_df, I,
        test_cons_neighbors, test_blind_neighbors,
        test_ents, train_ents,
    )
    blind_pairs_df = connect_blindtype_to_deviation(
        test_df, train_df, df_clean, test_ents, train_ents
    )

    # --- 8. Save CSVs ---
    test_df.to_csv(config.RESULTS_DIR / "deviation_results.csv", index=False)
    blind_pairs_df.to_csv(config.RESULTS_DIR / "blind_pairs_analysis.csv", index=False)
    print(f"Results saved to {config.RESULTS_DIR}")

    # --- 9. Visualizations ---
    print("\n--- 7. Visualizations ---")
    _run_visualizations(blind_pairs_df, test_df, config.RESULTS_DIR)

    print(f"\n====== PIPELINE COMPLETE ({args.mode.upper()}) ======")
    print(f"Artifacts: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
