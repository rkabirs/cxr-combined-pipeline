"""
compare_retrieval_modes.py

Loads results from all three retrieval modes (baseline CLIP, MoF, concept)
and produces a summary comparison table + visualizations.

Run from the cxr-blind/ directory:
    python compare_retrieval_modes.py

Outputs to: cxr-blind/comparison/
"""
import ast
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "comparison"
OUT.mkdir(exist_ok=True)

PALETTE = {"Baseline (CLIP)": "#4878CF", "MoF": "#6ACC65", "Concept": "#D65F5F"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_neighbor_list(val):
    """Parse a stringified Python list/tuple of ints from a CSV cell."""
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return []
    try:
        result = ast.literal_eval(str(val))
        return list(result)
    except Exception:
        return []


def load_deviation_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["blind_list"]      = df["blind_neighbors"].apply(parse_neighbor_list)
    df["consistent_list"] = df["consistent_neighbors"].apply(parse_neighbor_list)
    df["n_blind"]         = df["blind_list"].apply(len)
    df["n_consistent"]    = df["consistent_list"].apply(len)
    df["radgraph_deviation_full"] = pd.to_numeric(df["radgraph_deviation_full"], errors="coerce")
    df["missing_entities_full"]   = pd.to_numeric(df["missing_entities_full"],   errors="coerce")
    return df


def load_concept_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["blind_list"]      = df["blind_neighbors"].apply(parse_neighbor_list)
    df["consistent_list"] = df["consistent_neighbors"].apply(parse_neighbor_list)
    df["n_blind"]         = df["blind_list"].apply(len)
    df["n_consistent"]    = df["consistent_list"].apply(len)
    return df


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")

baseline_dev  = load_deviation_df(BASE / "baseline" / "deviation_results.csv")
baseline_bp   = pd.read_csv(BASE / "baseline" / "blind_pairs_analysis.csv")
mof_dev       = load_deviation_df(BASE / "mof"      / "deviation_results.csv")
mof_bp        = pd.read_csv(BASE / "mof"      / "blind_pairs_analysis.csv")
concept_ret   = load_concept_df(BASE / "concept"   / "retrieval_results.csv")

n_images = len(baseline_dev)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def mode_summary(label, dev_df, bp_df):
    total_blind      = bp_df is not None and len(bp_df) or dev_df["n_blind"].sum()
    total_consistent = dev_df["n_consistent"].sum()
    blind_rate       = dev_df["n_blind"].mean() / 10          # k=10 neighbors
    consistent_rate  = dev_df["n_consistent"].mean() / 10
    mean_deviation   = dev_df["radgraph_deviation_full"].mean() if "radgraph_deviation_full" in dev_df else float("nan")
    median_deviation = dev_df["radgraph_deviation_full"].median() if "radgraph_deviation_full" in dev_df else float("nan")
    mean_missing     = dev_df["missing_entities_full"].mean()   if "missing_entities_full"   in dev_df else float("nan")
    type_counts      = bp_df["blind_type"].value_counts().to_dict() if bp_df is not None else {}
    return dict(
        label=label,
        n_images=len(dev_df),
        total_blind_pairs=int(dev_df["n_blind"].sum()),
        total_consistent_pairs=int(total_consistent),
        blind_rate=round(blind_rate, 4),
        consistent_rate=round(consistent_rate, 4),
        mean_radgraph_deviation=round(mean_deviation, 2),
        median_radgraph_deviation=round(median_deviation, 1),
        mean_missing_entities=round(mean_missing, 2),
        type1=type_counts.get("Type 1", 0),
        type2=type_counts.get("Type 2", 0),
        type3=type_counts.get("Type 3", 0),
    )


def concept_summary(label, df):
    return dict(
        label=label,
        n_images=len(df),
        total_blind_pairs=int(df["n_blind"].sum()),
        total_consistent_pairs=int(df["n_consistent"].sum()),
        blind_rate=round(df["n_blind"].mean() / 10, 4),
        consistent_rate=round(df["n_consistent"].mean() / 10, 4),
        mean_radgraph_deviation=float("nan"),
        median_radgraph_deviation=float("nan"),
        mean_missing_entities=float("nan"),
        type1=0, type2=0, type3=0,
    )


rows = [
    mode_summary("Baseline (CLIP)", baseline_dev, baseline_bp),
    mode_summary("MoF",             mof_dev,      mof_bp),
    concept_summary("Concept",      concept_ret),
]
summary = pd.DataFrame(rows).set_index("label")

print("\n=== Summary Table ===")
print(summary.to_string())
summary.to_csv(OUT / "summary_table.csv")
print(f"\nSaved {OUT / 'summary_table.csv'}")


# ---------------------------------------------------------------------------
# Figure 1 — Blind pair and consistent pair counts
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
modes  = list(summary.index)
colors = [PALETTE[m] for m in modes]
x      = np.arange(len(modes))
w      = 0.5

ax = axes[0]
bars = ax.bar(x, summary["total_blind_pairs"], width=w, color=colors)
for bar, val in zip(bars, summary["total_blind_pairs"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
            f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel("Total blind pairs"); ax.set_title("Total Blind Pairs by Mode")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
bars = ax.bar(x, summary["total_consistent_pairs"], width=w, color=colors)
for bar, val in zip(bars, summary["total_consistent_pairs"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel("Total consistent pairs"); ax.set_title("Total Consistent Pairs by Mode")
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Blind vs. Consistent Retrieval Counts Across Modes", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig1_pair_counts.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig1_pair_counts.png'}")


# ---------------------------------------------------------------------------
# Figure 2 — Blind pair rate and consistent pair rate (per query)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
bar_w = 0.32
xi    = np.arange(len(modes))

blind_rates      = summary["blind_rate"].values * 100
consistent_rates = summary["consistent_rate"].values * 100

b1 = ax.bar(xi - bar_w / 2, blind_rates,      bar_w, label="Blind rate",      color=colors, alpha=0.9)
b2 = ax.bar(xi + bar_w / 2, consistent_rates, bar_w, label="Consistent rate",
            color=colors, alpha=0.4, hatch="//")

for bar, val in zip(b1, blind_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
for bar, val in zip(b2, consistent_rates):
    if val > 0.1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(xi); ax.set_xticklabels(modes)
ax.set_ylabel("% of k=10 neighbors")
ax.set_title("Blind and Consistent Neighbor Rates per Query")
solid   = mpatches.Patch(color="grey", alpha=0.9, label="Blind rate")
hatched = mpatches.Patch(facecolor="grey", alpha=0.4, hatch="//", label="Consistent rate")
ax.legend(handles=[solid, hatched])
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig2_neighbor_rates.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig2_neighbor_rates.png'}")


# ---------------------------------------------------------------------------
# Figure 3 — RadGraph deviation distributions (Baseline vs MoF)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
ax = axes[0]
data_box  = [baseline_dev["radgraph_deviation_full"].dropna().values,
             mof_dev["radgraph_deviation_full"].dropna().values]
bp = ax.boxplot(data_box, patch_artist=True, labels=["Baseline (CLIP)", "MoF"],
                medianprops=dict(color="black", linewidth=1.5))
for patch, color in zip(bp["boxes"], [PALETTE["Baseline (CLIP)"], PALETTE["MoF"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("RadGraph deviation (full neighbors)")
ax.set_title("RadGraph Deviation Distribution")
ax.spines[["top", "right"]].set_visible(False)

# CDF
ax = axes[1]
for label, df, color in [
    ("Baseline (CLIP)", baseline_dev, PALETTE["Baseline (CLIP)"]),
    ("MoF",             mof_dev,      PALETTE["MoF"]),
]:
    vals = np.sort(df["radgraph_deviation_full"].dropna().values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, label=label, linewidth=2)

ax.set_xlabel("RadGraph deviation score"); ax.set_ylabel("CDF")
ax.set_title("CDF of RadGraph Deviation (full neighbors)")
ax.legend(); ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

ax.text(0.98, 0.05,
        "Concept mode: RadGraph\nanalysis not yet computed",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="grey", style="italic")

plt.suptitle("RadGraph Deviation: Baseline vs MoF", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig3_deviation_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig3_deviation_distribution.png'}")


# ---------------------------------------------------------------------------
# Figure 4 — Missing entity rates
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mean missing entities per image
ax = axes[0]
means  = [baseline_dev["missing_entities_full"].mean(),
          mof_dev["missing_entities_full"].mean(),
          float("nan")]
labels = modes
bars   = ax.bar(np.arange(len(labels)), means, color=colors, width=0.5)
for bar, val in zip(bars, means):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.1,
                "N/A", ha="center", va="bottom", fontsize=8, color="grey")
ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("Mean missing entities per image")
ax.set_title("Mean Missing RadGraph Entities\n(vs. full neighbor set)")
ax.spines[["top", "right"]].set_visible(False)

# Missing entity distribution (violin, Baseline vs MoF)
ax = axes[1]
data_v = [baseline_dev["missing_entities_full"].dropna().values,
          mof_dev["missing_entities_full"].dropna().values]
vp = ax.violinplot(data_v, positions=[0, 1], showmedians=True)
for i, (patch, color) in enumerate(zip(vp["bodies"],
                                        [PALETTE["Baseline (CLIP)"], PALETTE["MoF"]])):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline (CLIP)", "MoF"])
ax.set_ylabel("Missing entities count")
ax.set_title("Missing Entity Distribution (full neighbors)")
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Missing RadGraph Entities Across Modes", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig4_missing_entities.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig4_missing_entities.png'}")


# ---------------------------------------------------------------------------
# Figure 5 — Blind type breakdown (Baseline vs MoF, stacked)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

type_data = {
    "Baseline (CLIP)": baseline_bp["blind_type"].value_counts(),
    "MoF":             mof_bp["blind_type"].value_counts(),
}

type_colors = {"Type 1": "#4878CF", "Type 2": "#6ACC65", "Type 3": "#D65F5F"}
x_pos = np.arange(len(type_data))
bottoms = np.zeros(len(type_data))

for btype, color in type_colors.items():
    vals = np.array([type_data[m].get(btype, 0) for m in type_data])
    bars = ax.bar(x_pos, vals, bottom=bottoms, label=btype, color=color, width=0.5)
    for bar, v, bot in zip(bars, vals, bottoms):
        if v > 200:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bot + v / 2,
                    f"{v:,}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks(x_pos); ax.set_xticklabels(list(type_data.keys()))
ax.set_ylabel("Number of blind pairs")
ax.set_title("Blind Pair Type Distribution\n(Concept: type labels not yet computed)")
ax.legend(title="Blind type", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig5_blind_type_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig5_blind_type_distribution.png'}")


# ---------------------------------------------------------------------------
# Figure 6 — Deviation vs. blind pair count scatter (per pathology, MoF)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (label, bp_df, dev_df, color) in zip(axes, [
    ("Baseline (CLIP)", baseline_bp, baseline_dev, PALETTE["Baseline (CLIP)"]),
    ("MoF",             mof_bp,      mof_dev,      PALETTE["MoF"]),
]):
    # merge blind pair count per uid with deviation score
    bp_counts = bp_df.groupby("query_uid").size().reset_index(name="blind_count")
    dev_sub   = dev_df[["uid", "radgraph_deviation_full", "missing_entities_full"]].copy()
    dev_sub["uid"] = dev_sub["uid"].astype(str)
    bp_counts["query_uid"] = bp_counts["query_uid"].astype(str)
    merged = bp_counts.merge(dev_sub, left_on="query_uid", right_on="uid", how="inner")
    ax.scatter(merged["blind_count"], merged["radgraph_deviation_full"],
               alpha=0.15, s=8, color=color)
    # trend line
    mask = merged["radgraph_deviation_full"].notna()
    if mask.sum() > 10:
        z = np.polyfit(merged.loc[mask, "blind_count"],
                       merged.loc[mask, "radgraph_deviation_full"], 1)
        xs = np.linspace(merged["blind_count"].min(), merged["blind_count"].max(), 100)
        ax.plot(xs, np.polyval(z, xs), color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Blind pair count per query")
    ax.set_ylabel("RadGraph deviation score")
    ax.set_title(f"{label}: Blind Count vs Deviation")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Do More Blind Pairs Correlate with Higher Deviation?", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig6_blind_count_vs_deviation.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig6_blind_count_vs_deviation.png'}")


# ---------------------------------------------------------------------------
# Figure 7 — Per-pathology blind pair rates (Baseline vs MoF)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))

path_counts = {}
for label, bp_df in [("Baseline (CLIP)", baseline_bp), ("MoF", mof_bp)]:
    vc = bp_df["primary_pathology"].value_counts()
    path_counts[label] = vc

all_paths = sorted(set(list(path_counts["Baseline (CLIP)"].index) +
                        list(path_counts["MoF"].index)))
x_vals = np.arange(len(all_paths))
w      = 0.38

ax.bar(x_vals - w / 2,
       [path_counts["Baseline (CLIP)"].get(p, 0) for p in all_paths],
       width=w, label="Baseline (CLIP)", color=PALETTE["Baseline (CLIP)"], alpha=0.85)
ax.bar(x_vals + w / 2,
       [path_counts["MoF"].get(p, 0) for p in all_paths],
       width=w, label="MoF", color=PALETTE["MoF"], alpha=0.85)

ax.set_xticks(x_vals)
ax.set_xticklabels(all_paths, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Blind pair count")
ax.set_title("Blind Pairs per Primary Pathology (Baseline vs MoF)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig7_blind_pairs_by_pathology.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'fig7_blind_pairs_by_pathology.png'}")


# ---------------------------------------------------------------------------
# Print formatted summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 68)
print(f"{'Metric':<35} {'Baseline':>10} {'MoF':>10} {'Concept':>10}")
print("=" * 68)
rows_display = [
    ("Images",                   n_images,                   n_images,               n_images),
    ("Total blind pairs",        summary.loc["Baseline (CLIP)","total_blind_pairs"],
                                 summary.loc["MoF","total_blind_pairs"],
                                 summary.loc["Concept","total_blind_pairs"]),
    ("Total consistent pairs",   summary.loc["Baseline (CLIP)","total_consistent_pairs"],
                                 summary.loc["MoF","total_consistent_pairs"],
                                 summary.loc["Concept","total_consistent_pairs"]),
    ("Blind rate (% of k=10)",   f"{summary.loc['Baseline (CLIP)','blind_rate']*100:.1f}%",
                                 f"{summary.loc['MoF','blind_rate']*100:.1f}%",
                                 f"{summary.loc['Concept','blind_rate']*100:.1f}%"),
    ("Consistent rate (% of k)", f"{summary.loc['Baseline (CLIP)','consistent_rate']*100:.1f}%",
                                 f"{summary.loc['MoF','consistent_rate']*100:.1f}%",
                                 f"{summary.loc['Concept','consistent_rate']*100:.1f}%"),
    ("Mean RadGraph deviation",  f"{summary.loc['Baseline (CLIP)','mean_radgraph_deviation']:.2f}",
                                 f"{summary.loc['MoF','mean_radgraph_deviation']:.2f}",
                                 "N/A"),
    ("Median RadGraph deviation",f"{summary.loc['Baseline (CLIP)','median_radgraph_deviation']:.1f}",
                                 f"{summary.loc['MoF','median_radgraph_deviation']:.1f}",
                                 "N/A"),
    ("Mean missing entities",    f"{summary.loc['Baseline (CLIP)','mean_missing_entities']:.2f}",
                                 f"{summary.loc['MoF','mean_missing_entities']:.2f}",
                                 "N/A"),
    ("Type 1 blind pairs",       f"{summary.loc['Baseline (CLIP)','type1']:,}",
                                 f"{summary.loc['MoF','type1']:,}",
                                 "N/A"),
    ("Type 2 blind pairs",       f"{summary.loc['Baseline (CLIP)','type2']:,}",
                                 f"{summary.loc['MoF','type2']:,}",
                                 "N/A"),
    ("Type 3 blind pairs",       f"{summary.loc['Baseline (CLIP)','type3']:,}",
                                 f"{summary.loc['MoF','type3']:,}",
                                 "N/A"),
]
for metric, b, m, c in rows_display:
    print(f"  {metric:<33} {str(b):>10} {str(m):>10} {str(c):>10}")
print("=" * 68)
print(f"\nAll figures saved to: {OUT}")
