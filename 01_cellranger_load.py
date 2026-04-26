#!/usr/bin/env python
"""
================================================================================
Sub 1.1 — Cell Ranger Output & Loading the Dataset
Script:   01_download.py
Series:   Multimodal Single-Cell Series (Series 1)
Dataset:  Single-cell multi-omics analysis of the immune response in COVID-19, Nat Med 27:904-916 (2021), E-MTAB-10026
URL:      https://covid19.cog.sanger.ac.uk/submissions/release1/
================================================================================

What this script does
---------------------
1. Sets up a project directory tree (RNA, CITE, VDJ, results, figures).
2. Downloads the multiome_covid19_2021 processed AnnData from the public Sanger
   COVID-19 Cell Atlas (the original ArrayExpress URLs no longer resolve).
3. Loads the AnnData and inspects what's inside.
4. Restores raw UMI counts from `adata.layers["raw"]` to `adata.X` (the
   author file ships with normalised values in .X).
5. Renames the authors' precomputed analysis (clusters, embeddings, QC
   metrics) with an `_author` suffix so our pipeline can recompute fresh
   versions side by side.
6. Builds a canonical, ordered `Severity` column and a colour palette.
7. Saves the prepared object as `00_raw.h5ad` — the entry checkpoint that
   every subsequent tutorial in the series loads from.

What it does NOT do
-------------------
- No cell or gene filtering.
- No QC computation (that lives in `02_qc.py` for Sub 1.2).
- No normalisation, HVG, PCA, or anything downstream.

Why
---
This is a hard separation of "data preparation" from "analysis" so that
every Sub 1.X script in the series can start from the same `00_raw.h5ad`
checkpoint without re-running the download.

Usage
-----
    python 01_download.py

Set the PROJECT path below to wherever you want the data tree to live.
~10 GB of free disk space recommended.

================================================================================
"""

# ============================================================================
# Imports + plot style
# ============================================================================

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white")

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

# Multiome Academy brand
MA_BLUE   = "#1A5276"
MA_BLUE2  = "#2E86C1"
MA_PURPLE = "#6C3483"

print(f"scanpy  {sc.__version__}")
print(f"anndata {ad.__version__}")
print(f"pandas  {pd.__version__}")
print(f"numpy   {np.__version__}")


# ============================================================================
# Project paths — edit PROJECT to your environment
# ============================================================================

PROJECT  = os.path.expanduser("~/multiome_covid19_2021")
RNA_DIR  = f"{PROJECT}/RNA"
CITE_DIR = f"{PROJECT}/CITE"
VDJ_DIR  = f"{PROJECT}/VDJ"
OUT_DIR  = f"{PROJECT}/results"
FIG_DIR  = f"{PROJECT}/figures"

for d in [RNA_DIR, CITE_DIR, VDJ_DIR, OUT_DIR, FIG_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

print("\nDirectory tree ready:")
for d in [PROJECT, RNA_DIR, CITE_DIR, VDJ_DIR, OUT_DIR, FIG_DIR]:
    print(f"  {d}")


# ============================================================================
# Download from Sanger COVID-19 Cell Atlas
# ----------------------------------------------------------------------------
# ArrayExpress files for E-MTAB-10026 have moved to BioStudies and the old
# direct URLs are broken. Sanger's Cellular Genetics object store hosts the
# authors' processed deliverable directly. Stable for years.
# ============================================================================

SANGER_BASE = "https://covid19.cog.sanger.ac.uk/submissions/release1"

FILES = {
    # Main processed object — GEX + CITE metadata + donor annotations
    "h5ad_main": (
        f"{SANGER_BASE}/haniffa21.processed.h5ad",
        f"{RNA_DIR}/haniffa21.processed.h5ad",
    ),
    # Critical-severity BCR-expanded subset (small, used in Series Part 4)
    "h5ad_bcr": (
        f"{SANGER_BASE}/BCR_expanded_Critical.h5ad",
        f"{VDJ_DIR}/BCR_expanded_Critical.h5ad",
    ),
    # SDRF (sample-level metadata) via BioStudies
    "sdrf": (
        "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/E-MTAB-10026.sdrf.txt",
        f"{RNA_DIR}/E-MTAB-10026.sdrf.txt",
    ),
}


def download_resumable(url, dest, min_expected_mb=1):
    """wget with resume (-c). Survives interrupted multi-GB downloads."""
    if os.path.exists(dest) and os.path.getsize(dest) > min_expected_mb * 1e6:
        size_gb = os.path.getsize(dest) / 1e9
        print(f"  ✓ exists:      {os.path.basename(dest)}  ({size_gb:,.2f} GB)")
        return
    print(f"  ↓ downloading: {os.path.basename(dest)}")
    print(f"    from: {url}")
    result = subprocess.run(
        ["wget", "-c", "--progress=dot:giga", "-O", dest, url],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for {url}")
    size_gb = os.path.getsize(dest) / 1e9
    print(f"  ✓ done:        {os.path.basename(dest)}  ({size_gb:,.2f} GB)")


# Order by size: small first so something works before the large pull
download_resumable(*FILES["sdrf"],      min_expected_mb=0.05)
download_resumable(*FILES["h5ad_bcr"],  min_expected_mb=10)
download_resumable(*FILES["h5ad_main"], min_expected_mb=500)

print(f"\n--- Contents of {RNA_DIR} ---")
for f in sorted(os.listdir(RNA_DIR)):
    fp = f"{RNA_DIR}/{f}"
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        size_str = f"{size/1e9:,.2f} GB" if size > 1e9 else f"{size/1e6:,.1f} MB"
        print(f"  {f}  ({size_str})")


# ============================================================================
# Load the AnnData object
# ============================================================================

H5AD_PATH = FILES["h5ad_main"][1]
print(f"\nLoading {H5AD_PATH} ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Inspect what's inside
print(f"\n--- Object structure ---")
print(f"  adata.X.dtype          : {adata.X.dtype}")
print(f"  adata.X is sparse      : {sparse.issparse(adata.X)}")
print(f"  adata.layers           : {list(adata.layers.keys())}")
print(f"  adata.obsm             : {list(adata.obsm.keys())}")
print(f"  adata.uns keys (first 10): {list(adata.uns.keys())[:10]}")


# ============================================================================
# Restore raw counts to .X
# ----------------------------------------------------------------------------
# The author h5ad stores normalised values in .X and raw counts in
# layers["raw"]. Every downstream Scanpy function (calculate_qc_metrics,
# normalize_total, log1p, highly_variable_genes, pca) expects raw counts
# in .X. Swap them now.
# ============================================================================

print("\nBefore swap:")
print(f"  .X max              = {adata.X.max():.2f}")
print(f"  .layers['raw'] max  = {adata.layers['raw'].max():.0f}")

adata.X = adata.layers["raw"].copy()
del adata.layers["raw"]

# Ensure sparse CSR (downstream tools expect this)
if not sparse.issparse(adata.X):
    adata.X = sparse.csr_matrix(adata.X)

# Verify integer-valued raw counts
x_sub = adata.X[:200, :1000].toarray()
print(f"\nAfter swap:")
print(f"  .X max              = {adata.X.max():.0f}")
print(f"  .X dtype            = {adata.X.dtype}")
print(f"  .X is integer-valued= {np.all(x_sub == x_sub.astype(int))}")


# ============================================================================
# Preserve the authors' analysis with `_author` suffix
# ----------------------------------------------------------------------------
# The author h5ad ships with full pipeline output already computed:
#   - full_clustering (51 cell types) and initial_clustering (18)
#   - X_pca, X_pca_harmony, X_umap
#   - QC metrics (n_genes, total_counts, pct_counts_mt, ...)
# We rename these with `_author` so our Sub 1.2+ pipeline can compute fresh
# versions without collision and we can build "our analysis vs the paper's"
# comparisons throughout the series.
# ============================================================================

# Cluster annotations
for src in ["full_clustering", "initial_clustering"]:
    if src in adata.obs.columns:
        adata.obs[f"{src}_author"] = adata.obs[src].copy()
        adata.obs.drop(columns=[src], inplace=True)
        print(f"  renamed obs:  {src} → {src}_author")

# QC metrics — Sub 1.2 will recompute these fresh
qc_cols = ["n_genes", "n_genes_by_counts", "total_counts",
           "total_counts_mt", "pct_counts_mt"]
for src in qc_cols:
    if src in adata.obs.columns:
        adata.obs[f"{src}_author"] = adata.obs[src].copy()
        adata.obs.drop(columns=[src], inplace=True)
        print(f"  renamed obs:  {src} → {src}_author")

# Embeddings
for src in ["X_pca", "X_pca_harmony", "X_umap"]:
    if src in adata.obsm:
        adata.obsm[f"{src}_author"] = adata.obsm[src].copy()
        del adata.obsm[src]
        print(f"  renamed obsm: {src} → {src}_author")

# Downstream-analysis objects in uns
for src in ["hvg", "leiden", "neighbors", "pca", "umap"]:
    if src in adata.uns:
        adata.uns[f"{src}_author"] = adata.uns[src]
        del adata.uns[src]
        print(f"  renamed uns:  {src} → {src}_author")


# ============================================================================
# Build canonical Severity column
# ----------------------------------------------------------------------------
# Status_on_day_collection_summary has 9 levels covering disease severity
# (Healthy → Critical) plus controls (Non_covid, LPS_90mins, LPS_10hours).
# Convert to ordered categorical so plots and groupby operations follow the
# clinically meaningful order.
# ============================================================================

SEVERITY_ORDER = [
    "Healthy", "Asymptomatic", "Mild",
    "Moderate", "Severe", "Critical",
    "Non_covid", "LPS_90mins", "LPS_10hours",
]

SEVERITY_PAL = {
    "Healthy":      "#2E86C1",   # blue
    "Asymptomatic": "#76D7C4",   # teal
    "Mild":         "#F7DC6F",   # yellow
    "Moderate":     "#F39C12",   # orange
    "Severe":       "#E74C3C",   # red
    "Critical":     "#922B21",   # dark red
    "Non_covid":    "#7F8C8D",   # gray
    "LPS_90mins":   "#BB8FCE",   # light purple
    "LPS_10hours":  "#6C3483",   # dark purple
}

adata.obs["Severity"] = pd.Categorical(
    adata.obs["Status_on_day_collection_summary"],
    categories=SEVERITY_ORDER,
    ordered=True,
)

adata.obs["Disease"] = adata.obs["Status"].copy()

adata.obs["Site"] = pd.Categorical(
    adata.obs["Site"],
    categories=["Ncl", "Cambridge", "Sanger"],
    ordered=False,
)

# Sanity-check
print("\n=== Severity counts (canonical column) ===")
print(adata.obs["Severity"].value_counts().reindex(SEVERITY_ORDER))
print(f"\nTotal cells: {adata.n_obs:,}")
print(f"Unmapped:    {adata.obs['Severity'].isna().sum():,}")

print("\n=== Disease × Site cross-tab ===")
print(pd.crosstab(adata.obs["Disease"], adata.obs["Site"]))


# ============================================================================
# Save 00_raw.h5ad
# ----------------------------------------------------------------------------
# Provenance metadata in adata.uns survives all downstream checkpoints.
# This is the entry point — every subsequent script in this series loads
# from here.
# ============================================================================

adata.uns["tutorial"] = {
    "series":     "Multimodal Single-Cell Series (Series 1)",
    "stage":      "raw",
    "paper":      "Single-cell multi-omics analysis of the immune response in COVID-19, Nat Med 27:904-916 (2021)",
    "accession":  "E-MTAB-10026",
    "source_url": FILES["h5ad_main"][0],
}

adata.uns["palettes"] = {
    "Severity": SEVERITY_PAL,
    "Sex":      {"Male": "#1A5276", "Female": "#A569BD"},
    "Site":     {"Ncl": "#1A5276", "Cambridge": "#2E86C1", "Sanger": "#6C3483"},
}
adata.uns["severity_order"] = SEVERITY_ORDER

out_path = f"{OUT_DIR}/00_raw.h5ad"
print(f"\nWriting {out_path} ...")
adata.write_h5ad(out_path, compression="gzip")
size_gb = os.path.getsize(out_path) / 1e9

print(f"\n✓ Saved: {out_path}")
print(f"  Size:  {size_gb:.2f} GB")
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print()
print("=" * 70)
print("Sub 1.1 download complete.")
print("Next: 02_qc.py — QC metrics, MAD audit, three figures.")
print("=" * 70)
