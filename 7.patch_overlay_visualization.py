import os
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
        df = pd.read_csv(f"{root_dir}/{model}_dist_to_%s_prototypes_wide.csv" % target)
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

        # value → RGBA → BGR (cv2)
        rgba = colormap(norm(v))
        color = tuple(int(255 * c) for c in rgba[:3][:3])  # RGB → BGR

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
    patch_scale_factor=2.0,  # 🔑 핵심
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
    patch_scale_factor=2.0,  # 핵심 보정
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
    patch_scale_factor=2.0,  # 핵심 보정
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

def plot_sample_overlay(
    sample,
    subtype,
    pdd_value,
    model_minmax,
    model_list,
    patch_size_list,
    coord_dir,
    svs_root_dir,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- find SVS automatically ----
    

    svs_subdir = os.path.join(svs_root_dir, subtype)
    svs_path = find_file_by_sample(svs_subdir, sample, suffix=".svs")         

    img, scale = load_svs_thumbnail(svs_path)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))


    for ax, model, patch_size in zip(axes, model_list, patch_size_list):
        # ---- find coords automatically ----
        coord_path = find_file_by_sample(coord_dir, sample, suffix=".npy")
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
    plt.close()
    
    
def patch_overlay_visualization(model_list, patch_size_list, base_dir, root_dir, svs_root_dir, coord_root_dir, out_dir, target):
    all_samples = collect_all_samples(root_dir, model_list, target)
    print(f"#Total unique samples: {len(all_samples)}")

    pdd_patch_dict = {}  # {model: {sample: np.ndarray}}
    
    pdd_col = 'PDD_%s' % target
    for model in model_list:
        df = pd.read_csv(os.path.join(root_dir, '%s_%s_pdd.csv' % (model, target)))
        model_dict_pdd = {}
        for sample, sub in df.groupby("Sample"):
            sub = sub.reset_index(drop=True)
            model_dict_pdd[sample] = sub[pdd_col].tolist()
        pdd_patch_dict[model] = model_dict_pdd
        
    pdd_patch_dict_base = {}  # {model: {sample: np.ndarray}}

    for model in model_list:
        df = pd.read_csv(os.path.join(base_dir, '%s_%s_pdd.csv' % (model, target)))

        model_dict_pdd = {}
        for sample, sub in df.groupby("Sample"):
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

    # 기준 모델 하나에서 meta 추출 (어느 모델이든 상관 없음)
    ref_df = pd.read_csv(f"{root_dir}/{model_list[0]}_dist_to_{target}_prototypes_wide.csv")
    sample_meta = (
        ref_df[["Sample", "Subtype"]]
        .drop_duplicates()
        .set_index("Sample")
    )

    os.makedirs(out_dir, exist_ok=True)

    for i, sample in enumerate(all_samples):
        if sample not in sample_meta.index:
            continue

        subtype = sample_meta.loc[sample, "Subtype"]

        print(f"[{i+1}/{len(all_samples)}] Processing {sample}")

        plot_sample_overlay(
            sample=sample,
            subtype=subtype,
            pdd_value = pdd_patch_dict,
            model_minmax = model_minmax,
            patch_size_list = patch_size_list,
            coord_root_dir=coord_root_dir,
            svs_root_dir=svs_root_dir,
            out_dir=out_dir,
        )

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
    return parser.parse_args()

def main():
    Argument = Parser_main()
    patch_overlay_visualization(Argument.pfm_list, Argument.patch_size_list, Argument.base_dir, Argument.root_dir, Argument.svs_root_dir, Argument.coord_root_dir, Argument.save_dir, Argument.target_column)

if __name__ == "__main__":
    main()