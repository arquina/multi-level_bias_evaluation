import os
import re
import numpy as np
import pandas as pd
import openslide
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import argparse

def find_center_cols(df, prefix="CENTER__"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        raise ValueError(f"Need >=2 columns starting with {prefix}. Found: {cols}")
    return cols

def compute_margin_min(df, center_cols, true_center_col="center", center_prefix="CENTER__"):
    """
    margin_min = d_true - min(d_other)
    negative => true center is closest
    """
    true_center_wide = center_prefix + df[true_center_col].astype(str)
    bad = ~true_center_wide.isin(center_cols)
    if bad.any():
        ex = df.loc[bad, true_center_col].value_counts().head(10)
        raise ValueError(
            "Some true centers don't match CENTER__ columns.\n"
            f"Examples (center value counts):\n{ex}"
        )

    D = df[center_cols].to_numpy(float)  # [N, K]
    idx = {c: i for i, c in enumerate(center_cols)}
    true_idx = np.array([idx[c] for c in true_center_wide], dtype=int)

    d_true = D[np.arange(len(D)), true_idx]

    mask = np.ones_like(D, dtype=bool)
    mask[np.arange(len(D)), true_idx] = False
    D_other = np.where(mask, D, np.nan)
    d_other_min = np.nanmin(D_other, axis=1)

    return d_true - d_other_min

def collect_all_samples(root_dir, model_list, target):
    samples = set()
    for model in model_list:
        df = pd.read_csv(os.path.join(root_dir, 'distance', '%s_dist_to_%s_prototypes_wide.csv' % (model, target)))
        samples.update(df["Sample"].unique().tolist())
    return sorted(list(samples))

def load_svs_thumbnail(svs_path, max_side=3000):
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.dimensions

    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))

    thumbnail = slide.get_thumbnail(new_size)
    thumbnail = np.array(thumbnail)

    return thumbnail, scale

def load_patch_coords(coord_path):
    coords = np.load(coord_path)
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    return coords

def overlay_patches_heatmap(
    ax,
    img,
    coords,
    values,
    scale,
    patch_size,
    cmap="RdBu_r",
    alpha=0.4,
    patch_scale_factor=2.0,
    vmin=None,
    vmax=None,
):
    """
    Draw patch-level heatmap overlay on thumbnail.

    coords: [(x, y), ...]  (level-0 top-left)
    values: [v1, v2, ...]  (scalar per patch)
    """

    assert len(coords) == len(values), "coords and values length mismatch"

    overlay = img.copy()

    # patch size on thumbnail
    w = int(patch_size * patch_scale_factor * scale)
    h = w

    # normalize values
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    for (x, y), v in zip(coords, values):
        x0 = int(x * scale)
        y0 = int(y * scale)
        x1 = x0 + w
        y1 = y0 + h

        # value ??RGBA ??BGR (cv2)
        rgba = colormap(norm(v))
        color = tuple(int(255 * c) for c in rgba[:3][:3])  # RGB ??BGR

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color,
            thickness=-1,
        )

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    ax.imshow(blended)
    ax.axis("off")
    
    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])

    # cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.15
    )

def overlay_patches(
    ax,
    img,
    coords,
    scale,
    patch_size,
    color=(255, 0, 0),
    alpha=0.2,
    patch_scale_factor=2.0,  # ?뵎 ?듭떖
):
    overlay = img.copy()

    # radius in thumbnail space
    r = int((patch_size * patch_scale_factor / 2) * scale)

    for (x, y) in coords:
        cx = int(x * scale)
        cy = int(y * scale)
        cv2.circle(overlay, (cx, cy), r, color, -1)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    ax.imshow(blended)
    ax.axis("off")

def overlay_patches_square(
    ax,
    img,
    coords,
    scale,
    patch_size,
    color=(0, 0, 255),
    alpha=0.25,
    patch_scale_factor=2.0,  # ?듭떖 蹂댁젙
):
    """
    Draw square patch overlays (top-left based) on thumbnail.
    """
    overlay = img.copy()

    w = int(patch_size * patch_scale_factor * scale)
    h = w

    for (x, y) in coords:
        x0 = int(x * scale)
        y0 = int(y * scale)
        x1 = x0 + w
        y1 = y0 + h

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color,
            thickness=-1,  # filled
        )

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    ax.imshow(blended)
    ax.axis("off")
    
def overlay_patches_double_square(
    ax,
    img,
    coords1,
    coords2,
    scale,
    patch_size,
    color1=(255, 0, 0),
    color2=(0, 0, 255),
    alpha=0.25,
    patch_scale_factor=2.0,  # ?듭떖 蹂댁젙
):
    """
    Draw square patch overlays (top-left based) on thumbnail.
    """
    overlay = img.copy()

    w = int(patch_size * patch_scale_factor * scale)
    h = w

    for (x, y) in coords1:
        x0 = int(x * scale)
        y0 = int(y * scale)
        x1 = x0 + w
        y1 = y0 + h

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color1,
            thickness=-1,  # filled
        )
        
    for (x, y) in coords2:
        x0 = int(x * scale)
        y0 = int(y * scale)
        x1 = x0 + w
        y1 = y0 + h

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color2,
            thickness=-1,  # filled
        )

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    ax.imshow(blended)
    ax.axis("off")


def find_file_by_sample(root_dir, sample_id, suffix):
    """
    Find a single file in root_dir that starts with sample_id and ends with suffix.
    Raises error if not exactly one match.
    """
    candidates = [
        f for f in os.listdir(root_dir)
        if f.startswith(sample_id) and f.endswith(suffix)
    ]
    if len(candidates) == 1:
        return os.path.join(root_dir, candidates[0])
    if len(candidates) == 0:
        return None
        # raise FileNotFoundError(f"No file found for sample '{sample_id}' in {root_dir}")
    if len(candidates) > 1:
        candidates2 = [
        f for f in os.listdir(root_dir)
        if f.startswith(sample_id+'.') and f.endswith(suffix)
        ]
        if len(candidates2) == 1:
            return os.path.join(root_dir, candidates2[0])
        else:
            raise RuntimeError(
                f"Multiple files found for sample '{sample_id}' in {root_dir}: {candidates}"
            )

def find_file_containing_sample(dirpath, filenames, sample_id, suffix):
    """
    Fallback for filenames that don't start with the sample id, e.g. SNUBH
    prefixes every file with the center name ('SNUBH_S12-1449-6.svs' for
    sample 'S12-1449'). Matches sample_id as a substring bounded by
    non-alphanumeric characters (or string edges) so 'S12-1449' doesn't
    also match a longer id like 'S12-14495'.
    """
    pattern = re.compile(r'(?:^|[^0-9A-Za-z])' + re.escape(sample_id) + r'(?:[^0-9A-Za-z]|$)')
    candidates = [f for f in filenames if f.endswith(suffix) and pattern.search(f)]
    if len(candidates) == 1:
        return os.path.join(dirpath, candidates[0])
    return None

def find_svs_path(svs_root_dir, center, subtype, sample):
    """
    Resolve a sample's .svs path. Internal cohort is nested center/subtype/
    (e.g. BORAMAE/KIRC/), but some centers (KHMC, SNUBH, GSH, GSH_histech)
    file everything under a single umbrella folder (e.g. KHMC/RCC/) instead
    of per-subtype-code folders, so the metadata subtype ('KIRP') doesn't
    match the actual directory name ('RCC'). Fall back to a recursive search
    under center/ before giving up (TCGA has no center subfolder at all, so
    that path is tried last, flat under svs_root_dir/subtype/). SNUBH also
    prefixes filenames with the center name, so within the walk we also try
    a substring match, not just startswith.
    """
    svs_subdir = os.path.join(svs_root_dir, center, subtype)
    if os.path.isdir(svs_subdir):
        path = find_file_by_sample(svs_subdir, sample, suffix=".svs")
        if path:
            return path

    center_dir = os.path.join(svs_root_dir, center)
    if os.path.isdir(center_dir):
        for dirpath, _, filenames in os.walk(center_dir):
            path = find_file_by_sample(dirpath, sample, suffix=".svs")
            if path:
                return path
            path = find_file_containing_sample(dirpath, filenames, sample, suffix=".svs")
            if path:
                return path

    flat_subdir = os.path.join(svs_root_dir, subtype)
    if os.path.isdir(flat_subdir):
        path = find_file_by_sample(flat_subdir, sample, suffix=".svs")
        if path:
            return path

    return None

# Submission_dir model-folder naming differs from the pfm_list names used
# elsewhere (script 6, pdd.csv filenames). Map pfm_list name -> folder name.
MODEL_DIR_MAP = {
    "virchow": "virchow",
    "virchow2": "virchow2",
    "UNI": "uni_v1",
    "UNI2": "uni_v2",
    "GigaPath": "gigapath",
    "CONCH": "conch_v1",
}

def plot_sample_overlay(
    sample,
    subtype,
    center,
    pdd_value,
    model_minmax,
    model_list,
    patch_size_list,
    coord_root_dir,
    svs_root_dir,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- find SVS automatically ----
    svs_path = find_svs_path(svs_root_dir, center, subtype, sample)
    if svs_path is None:
        raise FileNotFoundError(
            f"No .svs found for sample '{sample}' (center={center}, subtype={subtype}) under {svs_root_dir}"
        )

    img, scale = load_svs_thumbnail(svs_path)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))


    for ax, model, patch_size in zip(axes, model_list, patch_size_list):
        # ---- find coords automatically ----
        # Same center-nested-vs-flat ambiguity as svs above; each model can also
        # have a different patch_size, so this must be resolved per-model rather
        # than from one shared coord_dir.
        model_dirname = MODEL_DIR_MAP.get(model, model)
        coord_dir = os.path.join(coord_root_dir, center, model_dirname, str(patch_size), subtype, "coords")
        if not os.path.isdir(coord_dir):
            coord_dir = os.path.join(coord_root_dir, model_dirname, str(patch_size), subtype, "coords")
        coord_path = find_file_by_sample(coord_dir, sample, suffix=".npy")
        if coord_path is None and os.path.isdir(coord_dir):
            # SNUBH prefixes every filename with the center name
            # ('SNUBH_S12-1449-6.npy'), which a plain startswith match misses.
            coord_path = find_file_containing_sample(coord_dir, os.listdir(coord_dir), sample, suffix=".npy")
        if coord_path is None:
            raise FileNotFoundError(f"No .npy coords found for sample '{sample}' in {coord_dir}")
        coords_all = load_patch_coords(coord_path)

        vmin, vmax = model_minmax[model]

        overlay_patches_heatmap(
            ax=ax,
            img=img,
            coords=coords_all,        
            values=pdd_value[model][sample],        
            scale=scale,
            patch_size=patch_size,
            cmap="RdBu_r",
            alpha=0.5,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(model)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{sample}_tail_overlay.png"),
        dpi=200,
    )
    plt.savefig(
        os.path.join(out_dir, f"{sample}_tail_overlay.svg"),
        dpi=200,
    )
    plt.close()
    
    
def patch_overlay_visualization(model_list, patch_size_list, base_dir, root_dir, svs_root_dir, coord_root_dir, out_dir, target, max_samples=None, sample_list=None):
    all_samples = collect_all_samples(root_dir, model_list, target)
    print(f"#Total unique samples: {len(all_samples)}")
    if sample_list:
        missing = [s for s in sample_list if s not in all_samples]
        if missing:
            print(f"#WARNING: requested samples not found in this root_dir: {missing}")
        all_samples = [s for s in sample_list if s in all_samples]
        print(f"#Restricted to {len(all_samples)} requested samples")
    elif max_samples is not None:
        all_samples = all_samples[:max_samples]
        print(f"#Testing with first {len(all_samples)} samples")

    pdd_patch_dict = {}  # {model: {sample: np.ndarray}}

    # pdd.csv files use lowercase 'sample' and a capitalized 'PDD_Center'-style
    # column (target.capitalize()), unlike the *_dist_to_*_wide.csv files below
    # which use 'Sample'/'Subtype'. Both file types live in subfolders of
    # root_dir/base_dir ('pdd' and 'distance'), not directly inside them.
    pdd_col = 'PDD_%s' % target.capitalize()
    for model in model_list:
        df = pd.read_csv(os.path.join(root_dir, 'pdd', '%s_%s_pdd.csv' % (model, target)))
        model_dict_pdd = {}
        for sample, sub in df.groupby("sample"):
            sub = sub.reset_index(drop=True)
            model_dict_pdd[sample] = sub[pdd_col].tolist()
        pdd_patch_dict[model] = model_dict_pdd

    pdd_patch_dict_base = {}  # {model: {sample: np.ndarray}}

    for model in model_list:
        df = pd.read_csv(os.path.join(base_dir, 'pdd', '%s_%s_pdd.csv' % (model, target)))

        model_dict_pdd = {}
        for sample, sub in df.groupby("sample"):
            sub = sub.reset_index(drop=True)
            model_dict_pdd[sample] = sub[pdd_col].tolist()
        pdd_patch_dict_base[model] = model_dict_pdd

    model_minmax = {
        model: (
            min(x for lst in model_dict.values() for x in lst),
            max(x for lst in model_dict.values() for x in lst)
        )
        for model, model_dict in pdd_patch_dict_base.items()
    }

    # 湲곗? 紐⑤뜽 ?섎굹?먯꽌 meta 異붿텧 (?대뒓 紐⑤뜽?대뱺 ?곴? ?놁쓬)
    ref_df = pd.read_csv(os.path.join(root_dir, 'distance', f"{model_list[0]}_dist_to_{target}_prototypes_wide.csv"))
    sample_meta = (
        ref_df[["Sample", "Subtype", "Center"]]
        .drop_duplicates()
        .set_index("Sample")
    )

    os.makedirs(out_dir, exist_ok=True)

    for i, sample in enumerate(all_samples):
        if sample not in sample_meta.index:
            continue

        subtype = sample_meta.loc[sample, "Subtype"]
        center = sample_meta.loc[sample, "Center"]

        print(f"[{i+1}/{len(all_samples)}] Processing {sample}")

        try:
            plot_sample_overlay(
                sample=sample,
                subtype=subtype,
                center=center,
                pdd_value=pdd_patch_dict,
                model_minmax=model_minmax,
                model_list=model_list,
                patch_size_list=patch_size_list,
                coord_root_dir=coord_root_dir,
                svs_root_dir=svs_root_dir,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"#ERROR on {sample}: {e}")
            continue

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--pfm_list", nargs = "+", default = [], help = 'PFM list for comparison', type = str)
    parser.add_argument("--patch_size_list", nargs = "+", default = [], help = 'patch size list for PFMs', type = int)
    parser.add_argument("--base_dir", help = 'base_dir of analysis', type = str, required = False)
    parser.add_argument("--root_dir", help = 'root_dir of analysis', type = str, required = False)
    parser.add_argument("--svs_root_dir", help = 'Directory with svs files', type = str, required = False)
    parser.add_argument("--coord_root_dir", help = 'coord_dir of features', type = str, required = False)
    parser.add_argument("--save_dir", help = 'Directory to save the feature',type = str, required = False)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--max_samples", default = None, type = int, help = 'Limit to first N samples, for a quick test run')
    parser.add_argument("--sample_list", nargs = "+", default = None, help = 'Restrict to exactly these sample IDs (overrides --max_samples)', type = str)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    patch_overlay_visualization(Argument.pfm_list, Argument.patch_size_list, Argument.base_dir, Argument.root_dir, Argument.svs_root_dir, Argument.coord_root_dir, Argument.save_dir, Argument.target_column, Argument.max_samples, Argument.sample_list)

if __name__ == "__main__":
    main()