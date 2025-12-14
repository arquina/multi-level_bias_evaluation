#!/usr/bin/env python3
"""
02_rescue_analysis_clustering_montages_stats.py

Responsibilities
1) Using CSVs produced by Script 01, define patch-level rescue categories:
   - not_rescued
   - stainnorm_rescued
   - canceronly_rescued
   - both_rescued
2) Compute per-WSI category ratios and draw boxplot (hue=category).
3) Representative patch clustering per category:
   - KMeans within category (on features)
   - pick top-4 patches nearest each cluster centroid
   - crop from WSI and save 2x2 montage per cluster
   - horizontally stack cluster montages into one row image per category

Notes
- Rescue categories are defined from all_consistent flags across variants:
    not_rescued         : ~o & ~s & ~c & ~b
    stainnorm_rescued   : ~o &  s & ~c &  b  (matches your earlier stricter definition) OR (as requested here) we keep:
                          (~o) & s & b  AND (~c) is optional; we implement the strict version below.
    canceronly_rescued  : ~o & ~s &  c &  b
    both_rescued        : ~o & ~s & ~c &  b
- If you want a different rule (e.g., stainnorm_rescued = ~o & s regardless of b),
  change compute_rescue_masks() accordingly.
"""

import os
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import openslide
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


DEFAULT_VARIANTS = ["original", "stainnorm", "canceronly", "canceronly_stainnorm"]
DEFAULT_MODELS   = ["virchow", "uni_v1", "gigapath", "virchow2", "uni_v2", "conch_v1"]
DEFAULT_PATCH_SIZE = {"virchow":"224", "uni_v1":"256", "gigapath":"256", "virchow2":"224", "uni_v2":"256", "conch_v1":"512"}

# Consistency is computed across these centers (must match column name prefixes)
DEFAULT_CENTERS = ["MSKCC", "NCI Urologic Oncology Branch", "MD Anderson Cancer Center"]
DEFAULT_RACES = ['asian', 'black or african american', 'white']

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clean_patient_name(s: str) -> str:
    return "".join(str(s).split())

def find_svs_path(svs_root_dir: str, sample: str, subtype: str) -> Optional[str]:
    cand_dir = os.path.join(svs_root_dir, "TCGA", subtype)
    if not os.path.isdir(cand_dir):
        return None
    hits = glob.glob(os.path.join(cand_dir, f"*{sample}*.svs"))
    return hits[0] if hits else None

def find_feature_file(dataset_root: str, feature_type: str, patch_size: str, subtype: str, sample: str) -> Optional[str]:
    d = os.path.join(dataset_root, "whole_cancer", feature_type, patch_size, subtype, "total_feature")
    if not os.path.isdir(d):
        return None
    hits = [p for p in os.listdir(d) if (sample in p and p.endswith((".pt", ".pth")))]
    return os.path.join(d, hits[0]) if hits else None

def load_coords(coord_root_dir: str, model: str, patch_size: str, subtype: str, sample: str) -> Optional[np.ndarray]:
    d = os.path.join(coord_root_dir, model, patch_size, subtype, "coords")
    if not os.path.isdir(d):
        return None
    hits = [p for p in os.listdir(d) if (sample in p and p.endswith(".npy"))]
    if not hits:
        return None
    return np.load(os.path.join(d, hits[0]))

def df_has_xy(df: pd.DataFrame) -> bool:
    return ("x" in df.columns) and ("y" in df.columns)

def load_csv(interpret_root: str, variant: str, model: str, sample: str) -> Optional[pd.DataFrame]:
    p = os.path.join(interpret_root, variant, "sample_result", model, f"{sample}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


# ---------------------------
# Rescue category masks
# ---------------------------
def compute_rescue_masks(df_o: pd.DataFrame, df_s: pd.DataFrame, df_c: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Strict rescue definition (consistent with your earlier category logic):
      not_rescued         : ~o & ~s & ~c & ~b
      stainnorm_rescued   : ~o &  s & ~c &  b
      canceronly_rescued  : ~o & ~s &  c &  b
      both_rescued        : ~o & ~s & ~c &  b
    """
    o = df_o["all_consistent"].fillna(False).to_numpy(dtype=bool)
    s = df_s["all_consistent"].fillna(False).to_numpy(dtype=bool)
    c = df_c["all_consistent"].fillna(False).to_numpy(dtype=bool)
    b = df_b["all_consistent"].fillna(False).to_numpy(dtype=bool)

    masks = {
        "not_rescued":        (~o) & (~s) & (~c) & (~b),
        "stainnorm_rescued":  (~o) & ( s) & (~c) & ( b),
        "canceronly_rescued": (~o) & (~s) & ( c) & ( b),
        "both_rescued":       (~o) & (~s) & (~c) & ( b),
    }
    return masks


# ---------------------------
# Representative patch montages
# ---------------------------
def _crop_patch(slide: openslide.OpenSlide, xy: Tuple[int,int], patch_size: int = 512, level: int = 0) -> Image.Image:
    x, y = map(int, xy)
    return slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")

def _make_2x2_montage(images: List[Image.Image], tile_size: Tuple[int,int] = (192,192)) -> Image.Image:
    W, H = tile_size[0]*2, tile_size[1]*2
    canvas = Image.new("RGB", (W, H), (255,255,255))
    tiles = []
    for i in range(4):
        if i < len(images):
            tiles.append(images[i].resize(tile_size))
        else:
            tiles.append(Image.new("RGB", tile_size, (255,255,255)))
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (tile_size[0], 0))
    canvas.paste(tiles[2], (0, tile_size[1]))
    canvas.paste(tiles[3], (tile_size[0], tile_size[1]))
    return canvas

def _append_text_below(img: Image.Image, text: str, text_height: int = 28) -> Image.Image:
    W, H = img.size
    out = Image.new("RGB", (W, H + text_height), (255,255,255))
    out.paste(img, (0,0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, H + 7), text, fill=(0,0,0), font=font)
    return out

def _hstack(panels: List[Image.Image], pad: int = 8, bg=(255,255,255)) -> Image.Image:
    if not panels:
        return Image.new("RGB", (400, 400), bg)
    H = max(p.size[1] for p in panels)
    W = sum(p.size[0] for p in panels) + pad*(len(panels)-1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for i, p in enumerate(panels):
        y = (H - p.size[1]) // 2
        out.paste(p, (x, y))
        x += p.size[0] + (pad if i < len(panels)-1 else 0)
    return out

def _top4_near_centroid(X: np.ndarray, subset_indices: np.ndarray, centroid: np.ndarray) -> List[int]:
    if len(subset_indices) <= 4:
        return subset_indices.tolist()
    d = pairwise_distances(X, centroid.reshape(1, -1)).ravel()
    order = np.argsort(d)
    return subset_indices[order[:4]].tolist()

def build_category_row_montage(
    category_name: str,
    mask: np.ndarray,              # [N]
    features: np.ndarray,          # [N, D]
    coords: np.ndarray,            # [N, 2] level0 coords
    svs_path: str,
    out_dir: str,
    patch_size_level0: int,
    n_clusters: int = 4,
    tile_size: Tuple[int,int] = (192,192),
    random_state: int = 0,
) -> Optional[str]:
    """
    Cluster patches inside a category and save:
      - per cluster: category__cluster_{k}.jpg
      - row image:   category__row.jpg
    Returns row image path (or None if empty).
    """
    ensure_dir(out_dir)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        blank = Image.new("RGB", (tile_size[0]*2, tile_size[1]*2), (255,255,255))
        panel = _append_text_below(blank, f"{category_name}: no patches")
        row_path = os.path.join(out_dir, f"{category_name}__row.jpg")
        panel.save(row_path, quality=95)
        return row_path

    X = features[idxs]
    k = min(n_clusters, len(idxs))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    slide = openslide.OpenSlide(svs_path)
    cluster_panels = []
    rows = []
    try:
        for c in range(k):
            sub_local = np.where(labels == c)[0]
            sub_idxs = idxs[sub_local]
            rows.append({"cluster": int(c), "count": int(len(sub_idxs))})

            chosen = _top4_near_centroid(X[sub_local], sub_idxs, centers[c])

            imgs = []
            for ridx in chosen:
                xy = (int(coords[ridx, 0]), int(coords[ridx, 1]))
                imgs.append(_crop_patch(slide, xy, patch_size=patch_size_level0, level=0))

            montage = _make_2x2_montage(imgs, tile_size=tile_size)
            panel = _append_text_below(montage, f"{category_name} | cluster {c} (n={len(sub_idxs)})")
            panel_path = os.path.join(out_dir, f"{category_name}__cluster_{c}.jpg")
            panel.save(panel_path, quality=95)
            cluster_panels.append(panel)

        row_img = _hstack(cluster_panels, pad=8)
        row_path = os.path.join(out_dir, f"{category_name}__row.jpg")
        row_img.save(row_path, quality=95)

        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{category_name}__cluster_counts.csv"), index=False)
        return row_path
    finally:
        slide.close()


# ---------------------------
# Stats + plots
# ---------------------------
def collect_rescue_stats_for_all(
    meta_df: pd.DataFrame,
    interpret_root: str,
    variants: List[str],
    models: List[str],
) -> pd.DataFrame:
    """
    Per WSI, model: compute ratio of each rescue category.
    """
    rows = []
    v_o, v_s, v_c, v_b = variants

    for _, r in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Collect rescue stats"):
        sample = r["Patient"]
        for m in models:
            df_o = load_csv(interpret_root, v_o, m, sample)
            df_s = load_csv(interpret_root, v_s, m, sample)
            df_c = load_csv(interpret_root, v_c, m, sample)
            df_b = load_csv(interpret_root, v_b, m, sample)
            if any(x is None for x in [df_o, df_s, df_c, df_b]):
                continue

            # Align lengths defensively
            n = min(len(df_o), len(df_s), len(df_c), len(df_b))
            df_o = df_o.iloc[:n].copy()
            df_s = df_s.iloc[:n].copy()
            df_c = df_c.iloc[:n].copy()
            df_b = df_b.iloc[:n].copy()

            masks = compute_rescue_masks(df_o, df_s, df_c, df_b)
            for cat, mask in masks.items():
                rows.append({
                    "sample": sample,
                    "model": m,
                    "category": cat,
                    "ratio": float(mask.mean()) if len(mask) else np.nan,
                })
    return pd.DataFrame(rows)

def plot_rescue_boxplot(df: pd.DataFrame, models: List[str], categories: List[str], out_png: str):
    """
    Matplotlib boxplot grouped by model, colored by rescue category.
    """
    ensure_dir(os.path.dirname(out_png))

    data = {(m, c): [] for m in models for c in categories}
    for _, r in df.iterrows():
        if r["model"] in models and r["category"] in categories:
            data[(r["model"], r["category"])].append(r["ratio"])

    fig = plt.figure(figsize=(max(10, 1.8*len(models)), 6), dpi=160)
    ax = plt.gca()

    group_w = 0.8
    nC = len(categories)
    offsets = np.linspace(-group_w/2, group_w/2, nC, endpoint=False) + (group_w/nC)/2
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(categories)}

    positions, box_data = [], []
    for i, m in enumerate(models):
        base = i + 1
        for j, c in enumerate(categories):
            positions.append(base + offsets[j])
            box_data.append(data[(m, c)])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=(group_w/nC)*0.9,
        patch_artist=True,
        showfliers=False,
    )

    for k, patch in enumerate(bp["boxes"]):
        cat = categories[k % nC]
        patch.set_facecolor(cmap[cat])
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("Patch ratio")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    handles = [plt.Line2D([0],[0], color=cmap[c], lw=10, alpha=0.7) for c in categories]
    ax.legend(handles, categories, title="category", frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", help="Path to metadata.csv", default = "/mnt/disk1/Kidney/Submission_dir/metadata.csv")
    parser.add_argument("--svs_root_dir", help="Root for svs data (contains TCGA/{subtype})", default = '/mnt/disk3/svs_data/')
    parser.add_argument("--dataset_root", help="Root for Prototype dataset (contains whole_cancer/...)", default = '/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype')
    parser.add_argument("--coord_root_dir", help="Root for coords (model/patch_size/subtype/coords)", default = "/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/whole_cancer/")
    parser.add_argument("--interpret_root", help="Output root (interpretation)", default = "/mnt/disk1/Kidney/final_subission_dir/figure5/")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    help="Must be exactly 4 variants in order: original stainnorm canceronly canceronly_stainnorm")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--patch_size_json", default="", help="Optional JSON string or JSON file path for patch_size dict")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=192)
    parser.add_argument("--patch_size_multiplier", type=int, default=2)
    parser.add_argument("--skip_stats", action="store_true")
    parser.add_argument("--skip_montage", action="store_true")
    args = parser.parse_args()

    if len(args.variants) != 4:
        raise ValueError("variants must contain exactly 4 items: original stainnorm canceronly canceronly_stainnorm")

    patch_size_map = dict(DEFAULT_PATCH_SIZE)
    if args.patch_size_json:
        if os.path.exists(args.patch_size_json):
            with open(args.patch_size_json, "r") as f:
                patch_size_map.update(json.load(f))
        else:
            patch_size_map.update(json.loads(args.patch_size_json))

    meta = pd.read_csv(args.metadata_csv)
    meta = meta.drop_duplicates(subset=["Patient"]).copy()
    meta["Patient"] = meta["Patient"].astype(str).map(clean_patient_name)
    meta = meta[meta["subtype"].isin(["KIRC","KIRP","KICH"])].reset_index(drop=True)
    meta = meta[meta["subtype"].isin(["KIRC", "KIRP", "KICH"])].reset_index(drop=True)
    meta = meta[meta['center'].isin(DEFAULT_CENTERS)]
    meta = meta[meta['race'].isin(DEFAULT_RACES)]

    # ---- stats ----
    if not args.skip_stats:
        print("[STEP] rescue stats + boxplot")
        stats_df = collect_rescue_stats_for_all(meta, args.interpret_root, args.variants, args.models)
        stats_dir = os.path.join(args.interpret_root, "stats")
        ensure_dir(stats_dir)
        stats_csv = os.path.join(stats_dir, "rescue_ratios.csv")
        stats_df.to_csv(stats_csv, index=False)

        categories = ["not_rescued", "stainnorm_rescued", "canceronly_rescued", "both_rescued"]
        out_png = os.path.join(stats_dir, "boxplot_rescue_ratios.png")
        plot_rescue_boxplot(stats_df, args.models, categories, out_png)

        print(f"[DONE] rescue stats csv -> {stats_csv}")
        print(f"[DONE] rescue boxplot   -> {out_png}")

    # ---- representative patch clustering + montages ----
    if not args.skip_montage:
        print("[STEP] representative patch montages (category-wise clustering)")

        montage_root = os.path.join(args.interpret_root, "visualization", "rescue_montages")
        ensure_dir(montage_root)

        v_o, v_s, v_c, v_b = args.variants
        for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Montages"):
            sample = r["Patient"]
            subtype = str(r["subtype"])

            svs_path = find_svs_path(args.svs_root_dir, sample, subtype)
            if svs_path is None:
                continue

            for model in args.models:
                # Load CSVs (must exist)
                df_o = load_csv(args.interpret_root, v_o, model, sample)
                df_s = load_csv(args.interpret_root, v_s, model, sample)
                df_c = load_csv(args.interpret_root, v_c, model, sample)
                df_b = load_csv(args.interpret_root, v_b, model, sample)
                if any(x is None for x in [df_o, df_s, df_c, df_b]):
                    continue

                n = min(len(df_o), len(df_s), len(df_c), len(df_b))
                df_o = df_o.iloc[:n].copy()
                df_s = df_s.iloc[:n].copy()
                df_c = df_c.iloc[:n].copy()
                df_b = df_b.iloc[:n].copy()

                masks = compute_rescue_masks(df_o, df_s, df_c, df_b)

                # Load features for clustering:
                # We cluster using ORIGINAL feature space for the model (not stainnorm features),
                # because category definitions are across variants and original is a stable reference.
                patch_size = str(patch_size_map[model])
                feat_type = model  # original (no _stainnorm)
                feat_path = find_feature_file(args.dataset_root, feat_type, patch_size, subtype, sample)
                if feat_path is None:
                    continue

                feats = torch.load(feat_path, weights_only=True)
                if isinstance(feats, torch.Tensor):
                    feats_np = feats.cpu().numpy()
                else:
                    feats_np = np.asarray(feats)

                coords = load_coords(args.coord_root_dir, model, patch_size, subtype, sample)
                if coords is None:
                    continue

                # Align lengths
                n2 = min(len(feats_np), len(coords), n)
                feats_np = feats_np[:n2]
                coords = coords[:n2]
                for k in list(masks.keys()):
                    masks[k] = masks[k][:n2]

                out_dir = os.path.join(montage_root, sample, model)
                ensure_dir(out_dir)

                patch_size_level0 = int(patch_size) * args.patch_size_multiplier

                for cat, mask in masks.items():
                    build_category_row_montage(
                        category_name=cat,
                        mask=mask,
                        features=feats_np,
                        coords=coords,
                        svs_path=svs_path,
                        out_dir=out_dir,
                        patch_size_level0=patch_size_level0,
                        n_clusters=args.n_clusters,
                        tile_size=(args.tile_size, args.tile_size),
                        random_state=0,
                    )


if __name__ == "__main__":
    main()
