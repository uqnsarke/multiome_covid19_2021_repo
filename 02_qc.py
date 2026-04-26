#!/usr/bin/env python
"""
================================================================================
Sub 1.2 — Quality Control Audit
Script:   02_qc.py
Series:   Multimodal Single-Cell Series (Series 1)
Input:    {OUT_DIR}/00_raw.h5ad     (produced by Sub 1.1's 01_download.py)
Output:   {OUT_DIR}/01_qc_annotated.h5ad     + four figures in {FIG_DIR}
================================================================================

What this script does
---------------------
1. Loads the `00_raw.h5ad` checkpoint.
2. Computes fresh QC metrics (mitochondrial, ribosomal, hemoglobin
   percentages, transcriptional complexity).
3. Validates our metrics against the authors' precomputed values
   (Figure 2d — should be identical, since both use the same raw counts).
4. Visualises QC distributions per sample (Figure 2a).
5. Visualises cell counts per sample and per severity (Figure 2b).
6. Visualises QC distributions per severity group (Figure 2c).
7. Applies the median absolute deviation (MAD) outlier framework as an
   AUDIT — annotates flagged cells but does not filter them. The dataset
   has already been QC'd by the authors; this layer of analysis tells us
   whether the authors' filtering is concordant with a per-sample MAD
   approach, and where the residual outliers concentrate.
8. Saves `01_qc_annotated.h5ad` with QC flags added but all cells retained.

What it does NOT do
-------------------
- No cells removed (the authors already filtered; we audit).
- No doublet detection (deferred to a separate script).
- No normalisation, HVG, PCA (later Subs).

Usage
-----
    python 02_qc.py

================================================================================
"""

# ============================================================================
# Imports + plot style
# ============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.stats import median_abs_deviation as mad

warnings.filterwarnings("ignore")

PROJECT = os.path.expanduser("~/multiome_covid19_2021")
OUT_DIR = f"{PROJECT}/results"
FIG_DIR = f"{PROJECT}/figures"

plt.rcParams.update({
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

MA_BLUE   = "#1A5276"
MA_BLUE2  = "#2E86C1"
MA_PURPLE = "#6C3483"


def save(name):
    path = f"{FIG_DIR}/{name}"
    plt.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    print(f"✓ Saved: {path}")
    return path


# ============================================================================
# Load 00_raw.h5ad
# ============================================================================

adata = sc.read_h5ad(f"{OUT_DIR}/00_raw.h5ad")
print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

SEVERITY_ORDER = list(adata.uns["severity_order"])
SEVERITY_PAL   = dict(adata.uns["palettes"]["Severity"])
SEX_PAL        = dict(adata.uns["palettes"]["Sex"])
SITE_PAL       = dict(adata.uns["palettes"]["Site"])


# ============================================================================
# Compute fresh QC metrics
# ----------------------------------------------------------------------------
# We recompute rather than rely on the authors' columns. Three gene-set
# flags:
#   - mt   : mitochondrial (MT-)        → cell stress / death
#   - ribo : ribosomal (RPS/RPL)        → translation activity
#   - hb   : hemoglobin (HBB, HBA1...)  → red-blood-cell contamination
#
# Plus complexity (log10 genes / log10 UMIs) — a fourth axis that flags
# low-complexity cells (often dying or RBCs that survived lysis).
# ============================================================================

adata.var["mt"]   = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"]   = adata.var_names.str.contains(r"^HB[^(P)]", regex=True)

print("\nGene-set membership counts:")
for k in ["mt", "ribo", "hb"]:
    print(f"  {k:<6s} {adata.var[k].sum():>4d}")

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo", "hb"],
    percent_top=None,
    log1p=False,
    inplace=True,
)

adata.obs["log10_genes_per_umi"] = (
    np.log10(adata.obs["n_genes_by_counts"].clip(lower=1)) /
    np.log10(adata.obs["total_counts"].clip(lower=1))
)

print("\nOur QC metric summary:")
for c in ["n_genes_by_counts", "total_counts",
          "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb",
          "log10_genes_per_umi"]:
    d = adata.obs[c]
    print(f"  {c:<25s}  median={d.median():>8.1f}  "
          f"p01={d.quantile(0.01):>8.1f}  p99={d.quantile(0.99):>8.1f}")


# ============================================================================
# Figure 2d — Validate against authors' QC metrics
# ----------------------------------------------------------------------------
# Sanity check: our values should match the authors' (stored as *_author in
# obs) very closely, because both come from the same raw counts. Any
# disagreement flags an upstream gene-set or filtering mismatch we should
# investigate before trusting the rest of the QC.
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

pairs = [
    ("total_counts",      "total_counts_author",      "Total UMI counts"),
    ("n_genes_by_counts", "n_genes_by_counts_author", "Genes per cell"),
    ("pct_counts_mt",     "pct_counts_mt_author",     "% mitochondrial"),
]

for ax, (ours, theirs, title) in zip(axes, pairs):
    if theirs not in adata.obs.columns:
        ax.text(0.5, 0.5, f"{theirs}\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight="bold")
        continue

    x = adata.obs[theirs].values
    y = adata.obs[ours].values

    ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log", mincnt=1)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "--", color=MA_PURPLE, lw=1.5, alpha=0.7)

    mask = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.4f}", transform=ax.transAxes,
            va="top", fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xlabel(f"Authors' {title}", fontsize=11)
    ax.set_ylabel(f"Our {title}", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=MA_BLUE)

fig.suptitle("Figure 2d: Our QC metrics vs authors' (validation)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save("fig2d_qc_validation.png")


# ============================================================================
# Figure 2a — Per-sample QC distributions (horizontal boxplots, Blues gradient)
# ----------------------------------------------------------------------------
# Six panels share a y-axis: one row per sample, ordered by median UMI count
# (ascending). Boxes are coloured along a Blues gradient from low to high
# median, providing an immediate visual signal of which samples cluster
# together at each end of the distribution.
# ============================================================================

metrics = ["total_counts", "n_genes_by_counts",
           "pct_counts_mt", "pct_counts_ribo",
           "pct_counts_hb", "log10_genes_per_umi"]
titles = ["UMI counts", "Genes per cell",
          "% mitochondrial", "% ribosomal",
          "% hemoglobin", "Complexity (log10 genes/UMI)"]

# Order samples by median UMI count
sample_order = (adata.obs.groupby("sample_id", observed=True)["total_counts"]
                          .median().sort_values(ascending=True).index.tolist())

fig, axes = plt.subplots(1, 6, figsize=(20, 18), sharey=True)

for ax, metric, title in zip(axes, metrics, titles):
    medians = adata.obs.groupby("sample_id", observed=True)[metric].median()
    ranks = medians.rank(pct=True)
    cmap = plt.cm.Blues
    colors = {grp: cmap(0.25 + 0.70 * ranks[grp]) for grp in sample_order}

    for i, grp in enumerate(sample_order):
        vals = adata.obs.loc[adata.obs["sample_id"] == grp, metric].dropna()
        if len(vals) == 0:
            continue
        ax.boxplot(
            vals, positions=[i], vert=False, widths=0.65,
            patch_artist=True, notch=False,
            medianprops=dict(color="white", lw=1.8),
            boxprops=dict(facecolor=colors[grp], alpha=0.92,
                          edgecolor=colors[grp]),
            whiskerprops=dict(color=colors[grp], lw=0.8),
            capprops=dict(color=colors[grp], lw=0.8),
            flierprops=dict(marker=".", ms=1.2, alpha=0.12,
                            markerfacecolor=colors[grp],
                            markeredgecolor="none", linestyle="none"),
        )

    all_vals = adata.obs[metric].dropna()
    lo = max(all_vals.quantile(0.001), 0) if (
        metric.startswith("pct") or metric == "log10_genes_per_umi"
    ) else all_vals.quantile(0.001)
    hi = all_vals.quantile(0.999)
    ax.set_xlim(lo, hi)

    if metric in ("total_counts", "n_genes_by_counts"):
        ax.set_xscale("log")
    ax.set_yticks(range(len(sample_order)))
    ax.set_yticklabels(sample_order, fontsize=5.5)
    ax.set_xlabel(title, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", color=MA_BLUE)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.tick_params(axis="x", labelsize=9)

axes[0].set_ylabel(f"Sample (n={len(sample_order)}, sorted by median UMI)",
                   fontsize=11, fontweight="bold")
fig.suptitle(
    f"Figure 2a: Per-sample QC distributions "
    f"({adata.n_obs:,} cells, {len(sample_order)} samples, authors' QC-passed)",
    fontsize=13.5, fontweight="bold", y=0.995,
)
plt.tight_layout()
save("fig2a_qc_boxplot_persample.pdf")


# ============================================================================
# Figure 2b — Cell counts (per sample, per severity)
# ----------------------------------------------------------------------------
# Two panels:
#   Left  — bar chart of cells per sample, coloured by severity, sorted
#           within severity groups by cell count. Shows both the total
#           cell yield distribution and the severity grouping at a glance.
#   Right — total cells per severity group, annotated with donor counts.
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                         gridspec_kw={"width_ratios": [2.2, 1]})

# Left panel
sample_severity = (adata.obs.groupby("sample_id", observed=True)["Severity"]
                              .agg(lambda s: s.mode()[0]))
sample_counts = adata.obs["sample_id"].value_counts()

plot_df = (pd.DataFrame({"n_cells":  sample_counts,
                         "Severity": sample_severity})
             .reset_index(names="sample_id"))
plot_df["sev_rank"] = plot_df["Severity"].map(
    {s: i for i, s in enumerate(SEVERITY_ORDER)}
)
plot_df = plot_df.sort_values(["sev_rank", "n_cells"], ascending=[True, False])

ax = axes[0]
ax.bar(
    range(len(plot_df)), plot_df["n_cells"],
    color=[SEVERITY_PAL.get(s, "#888") for s in plot_df["Severity"]],
    edgecolor="white", linewidth=0.3,
)
ax.set_xticks([])
ax.set_xlabel(f"Sample (n={len(plot_df)}, ordered by severity then cell count)",
              fontsize=11)
ax.set_ylabel("Cells per sample", fontsize=11, fontweight="bold")
ax.set_title("Cells per sample (coloured by severity)",
             fontsize=12, fontweight="bold", color=MA_BLUE)
ax.axhline(plot_df["n_cells"].median(), color="black",
           lw=1, ls="--", alpha=0.5,
           label=f"median = {plot_df['n_cells'].median():,.0f}")
ax.legend(loc="upper right", fontsize=9, frameon=False)

# Severity colour spans below x-axis
x_start = 0
for sev in SEVERITY_ORDER:
    n = (plot_df["Severity"] == sev).sum()
    if n == 0:
        continue
    ax.axvspan(x_start - 0.5, x_start + n - 0.5,
               ymin=-0.06, ymax=-0.01,
               facecolor=SEVERITY_PAL[sev], alpha=0.9,
               clip_on=False)
    x_start += n

# Right panel
sev_totals = (adata.obs["Severity"].value_counts()
                     .reindex(SEVERITY_ORDER, fill_value=0))
sev_donors = (adata.obs.groupby("Severity", observed=True)["patient_id"]
                         .nunique().reindex(SEVERITY_ORDER, fill_value=0))

ax = axes[1]
y_pos = range(len(SEVERITY_ORDER))
ax.barh(y_pos, sev_totals.values,
        color=[SEVERITY_PAL[s] for s in SEVERITY_ORDER],
        edgecolor="white", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(SEVERITY_ORDER, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Total cells", fontsize=11)
ax.set_title("Cells per severity group",
             fontsize=12, fontweight="bold", color=MA_BLUE)
ax.grid(axis="x", alpha=0.3, linestyle=":")

for i, (cells, donors) in enumerate(zip(sev_totals.values, sev_donors.values)):
    ax.text(cells * 1.02, i, f" n={donors:,} donors",
            va="center", fontsize=9, fontweight="bold",
            color=SEVERITY_PAL[SEVERITY_ORDER[i]])

ax.set_xlim(0, sev_totals.max() * 1.22)

fig.suptitle(
    f"Figure 2b: Cell counts — "
    f"{adata.n_obs:,} cells, {adata.obs['patient_id'].nunique()} donors, "
    f"{len(plot_df)} samples",
    fontsize=13.5, fontweight="bold", y=1.02,
)
plt.tight_layout()
save("fig2b_cell_counts.pdf")


# ============================================================================
# Figure 2c — QC metrics by severity (boxplots, dashed median)
# ----------------------------------------------------------------------------
# Boxplot is the cleanest summary at this scale (647k cells). Whiskers at
# 1.5×IQR, fliers hidden — at this many cells, every box would have tens of
# thousands of fliers and the visualisation becomes unreadable. Median line
# dashed for emphasis.
# ============================================================================

metrics_c = ["total_counts", "n_genes_by_counts",
             "pct_counts_mt", "pct_counts_ribo",
             "pct_counts_hb", "log10_genes_per_umi"]
titles_c  = ["UMI counts (log)", "Genes per cell (log)",
             "% mitochondrial", "% ribosomal",
             "% hemoglobin", "Complexity"]
logy = [True, True, False, False, False, False]

present_sev = [s for s in SEVERITY_ORDER
               if s in adata.obs["Severity"].cat.categories
               and (adata.obs["Severity"] == s).any()]
pal_present = {s: SEVERITY_PAL[s] for s in present_sev}

fig, axes = plt.subplots(1, 6, figsize=(28, 5.5))

for ax, metric, title, is_log in zip(axes, metrics_c, titles_c, logy):
    data_per_group = [
        adata.obs.loc[adata.obs["Severity"] == s, metric].dropna().values
        for s in present_sev
    ]
    bp = ax.boxplot(
        data_per_group, positions=range(len(present_sev)),
        patch_artist=True, notch=False, widths=0.72,
        showfliers=False,
        medianprops=dict(color="black", lw=2.8, linestyle="--"),
        boxprops=dict(alpha=0.88),
        whiskerprops=dict(lw=1.6, linestyle="--", color="#333"),
        capprops=dict(lw=1.8, color="#333"),
    )

    for line in bp["medians"]:
        line.set_linestyle("--")
        line.set_linewidth(2.8)
        line.set_color("black")

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(pal_present[present_sev[i]])
        patch.set_edgecolor(pal_present[present_sev[i]])

    if is_log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(present_sev)))
    ax.set_xticklabels(present_sev, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(title, fontsize=10.5)
    ax.set_title(title, fontsize=12, fontweight="bold", color=MA_BLUE)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

fig.suptitle(
    f"Figure 2c: QC metrics by severity "
    f"({adata.n_obs:,} cells; whiskers = 1.5×IQR, fliers hidden for clarity)",
    fontsize=12.5, fontweight="bold", y=1.02,
)
plt.tight_layout()
save("fig2c_qc_by_severity.pdf")


# ============================================================================
# MAD audit — flag outliers without filtering
# ----------------------------------------------------------------------------
# The dataset has already been QC'd by the authors. Our job is to AUDIT
# their work using the median absolute deviation (MAD) framework applied
# per sample. Four criteria:
#
#   - high log1p(UMI)              ambient / multiplet risk
#   - high log1p(genes)            doublet / multiplet risk
#   - high % mitochondrial         dying / stressed cells
#   - low  log10_genes_per_umi     low-complexity cells
#
# Each criterion is stored in adata.obs as a boolean. NO CELLS ARE REMOVED.
# Learners see which cells the MAD framework considers marginal given the
# authors' QC.
# ============================================================================

def mad_outlier(series, nmads=3, higher_is_bad=True):
    """Flag values more than `nmads` MADs from the median."""
    m = np.median(series)
    d = mad(series)
    if d == 0:
        return np.zeros(len(series), dtype=bool)
    return (series > m + nmads * d).values if higher_is_bad \
        else (series < m - nmads * d).values


flag_umi   = np.zeros(adata.n_obs, dtype=bool)
flag_genes = np.zeros(adata.n_obs, dtype=bool)
flag_mt    = np.zeros(adata.n_obs, dtype=bool)
flag_cmplx = np.zeros(adata.n_obs, dtype=bool)

for sample, idx in adata.obs.groupby("sample_id", observed=True).groups.items():
    sub = adata.obs.loc[idx]
    in_idx = adata.obs.index.isin(idx)
    flag_umi[in_idx]   = mad_outlier(np.log1p(sub["total_counts"]))
    flag_genes[in_idx] = mad_outlier(np.log1p(sub["n_genes_by_counts"]))
    flag_mt[in_idx]    = mad_outlier(sub["pct_counts_mt"])
    flag_cmplx[in_idx] = mad_outlier(sub["log10_genes_per_umi"],
                                     higher_is_bad=False)

adata.obs["qc_flag_umi"]        = flag_umi
adata.obs["qc_flag_genes"]      = flag_genes
adata.obs["qc_flag_mt"]         = flag_mt
adata.obs["qc_flag_complexity"] = flag_cmplx
adata.obs["qc_flag_any"]        = (flag_umi | flag_genes |
                                   flag_mt  | flag_cmplx)

print("\n=== MAD audit (cells flagged per criterion, no filtering applied) ===")
print(f"  High log1p(UMI)        : {flag_umi.sum():>8,}  "
      f"({flag_umi.mean()*100:.2f}%)")
print(f"  High log1p(genes)      : {flag_genes.sum():>8,}  "
      f"({flag_genes.mean()*100:.2f}%)")
print(f"  High % MT              : {flag_mt.sum():>8,}  "
      f"({flag_mt.mean()*100:.2f}%)")
print(f"  Low complexity         : {flag_cmplx.sum():>8,}  "
      f"({flag_cmplx.mean()*100:.2f}%)")
print(f"  ANY flag (union)       : {adata.obs['qc_flag_any'].sum():>8,}  "
      f"({adata.obs['qc_flag_any'].mean()*100:.2f}%)")

print("\n=== Flagged cells by severity ===")
flag_by_sev = (adata.obs.groupby("Severity", observed=True)["qc_flag_any"]
                         .agg(["sum", "count", "mean"]))
flag_by_sev["mean"] = (flag_by_sev["mean"] * 100).round(2)
flag_by_sev.columns = ["n_flagged", "n_total", "pct_flagged"]
print(flag_by_sev.reindex(SEVERITY_ORDER).to_string())


# ============================================================================
# Save 01_qc_annotated.h5ad
# ============================================================================

adata.uns["tutorial"]["stage"] = "qc_annotated"
adata.uns["tutorial"]["qc_notes"] = {
    "framing":        "qc_check_preprocessed",
    "cells_flagged":  int(adata.obs["qc_flag_any"].sum()),
    "cells_retained": int(adata.n_obs),
    "comment":        "No filtering applied — QC flags are annotations only.",
}

out_path = f"{OUT_DIR}/01_qc_annotated.h5ad"
adata.write_h5ad(out_path, compression="gzip")
size_gb = os.path.getsize(out_path) / 1e9

print(f"\n✓ Saved: {out_path}  ({size_gb:.2f} GB)")
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  New obs columns: qc_flag_umi, qc_flag_genes, qc_flag_mt, "
      f"qc_flag_complexity, qc_flag_any")
print()
print("=" * 70)
print("Sub 1.2 QC audit complete.")
print("Next: Sub 1.3 — doublet detection, normalisation, HVG selection.")
print("=" * 70)
