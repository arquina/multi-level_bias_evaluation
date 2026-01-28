import os
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def stratified_split_meta(
    meta_df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
    subtype_col: str = "subtype",
    center_col: str = "center",
):
    """
    Split meta_df into train/test (7:3 default) while preserving subtype and center composition.
    Priority:
      (subtype, center) stratify -> subtype stratify -> random split.

    Returns: train_df, test_df, info_dict
    """
    df = meta_df.copy().reset_index(drop=True)

    # sanity
    assert 0 < test_size < 1
    assert subtype_col in df.columns, f"Missing column: {subtype_col}"
    assert center_col in df.columns, f"Missing column: {center_col}"

    # combined stratum label: "KIRC__CenterA"
    strata = df[subtype_col].astype(str) + "__" + df[center_col].astype(str)
    counts = strata.value_counts(dropna=False)

    # Helper: check if stratify is feasible.
    # sklearn requires each class in stratify to have at least 2 samples (practically),
    # and both train/test must be able to receive at least 1 sample from each stratum.
    def stratify_feasible(labels: pd.Series, test_size: float) -> bool:
        vc = labels.value_counts(dropna=False)
        if (vc < 2).any():
            return False
        n = len(labels)
        n_test = int(np.floor(n * test_size))
        n_train = n - n_test
        # each stratum needs at least 1 in train and 1 in test
        # => count >= 2 and also n_test >= number_of_strata and n_train >= number_of_strata (rough guard)
        k = vc.shape[0]
        if n_test < k or n_train < k:
            return False
        return True

    # Try (subtype, center) stratify first
    if stratify_feasible(strata, test_size):
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=strata,
        )
        mode = "stratify=subtype+center"
    else:
        # Fallback: subtype-only stratify
        subtype_labels = df[subtype_col].astype(str)
        if stratify_feasible(subtype_labels, test_size):
            train_idx, test_idx = train_test_split(
                df.index.to_numpy(),
                test_size=test_size,
                random_state=seed,
                stratify=subtype_labels,
            )
            mode = "stratify=subtype_only"
        else:
            # Last resort: random split
            train_idx, test_idx = train_test_split(
                df.index.to_numpy(),
                test_size=test_size,
                random_state=seed,
                stratify=None,
            )
            mode = "random_split"

    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    test_df  = df.loc[test_idx].copy().reset_index(drop=True)

    info = {
        "mode": mode,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "strata_counts": counts.to_dict(),
    }
    return train_df, test_df, info


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

def build_prototypes(
    X: np.ndarray,
    meta: pd.DataFrame,
    group_cols,
    normalize_after_mean: bool = True,
    normalize_X: bool = True,
):
    """
    Build prototypes (centroids) for groups defined by group_cols.

    - X: (N, D) slide embeddings
    - meta: metadata DataFrame aligned with X rows
    - group_cols: str or list[str] (e.g., "Center", ["Center","Subtype"])

    Normalization strategy (recommended):
      normalize_X=True -> L2 normalize slide embeddings before distance calculations
      normalize_before_mean=False -> do NOT unit-normalize before averaging (avoid weighting artifacts)
      normalize_after_mean=True -> L2 normalize the prototype centroids

    Returns:
      prototypes_df: DataFrame with one row per group and prototype vector stored separately
      protos: dict {group_key: prototype_vector (D,)}
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    assert X.shape[0] == len(meta), "X rows must match meta rows"

    X_use = X.copy()
    if normalize_X:
        X_use = l2_normalize(X_use, axis=1)

    protos = {}
    rows = []

    # groupby indices
    for key, idx in meta.groupby(group_cols).groups.items():
        idx = np.array(list(idx), dtype=int)
        Xg = X_use[idx]

        p = Xg.mean(axis=0)

        if normalize_after_mean:
            p = l2_normalize(p[None, :], axis=1)[0]

        protos[key] = p
        rows.append((*((key,) if not isinstance(key, tuple) else key), len(idx)))

    prototypes_df = pd.DataFrame(rows, columns=[*group_cols, "n_samples"])
    return prototypes_df, protos

def build_prototype(feature_root_dir, df, target_column, save_dir, feature_type, target_patch_size, data_type):
    data_save_dir = os.path.join(save_dir, 'prototype')
    os.makedirs(os.path.join(save_dir, 'prototype'), exist_ok= True)

    if feature_type == 'UNI':
        target_feature = 'uni_v1'
    elif feature_type == 'UNI2':
        target_feature = 'uni_v2'
    elif feature_type == 'GigaPath':
        target_feature = 'gigapath'
    elif feature_type == 'CONCH':
        target_feature = 'conch_v1'
    else:
        target_feature = feature_type

    if 'stainnorm' in data_type:
        feature_type += '_stainnorm'
        target_feature += '_stainnorm'
    
    if 'canceronly' in data_type:
        cancer_root = os.path.join(feature_root_dir, 'cancer_only', target_feature, target_patch_size)

    else:
        cancer_root = os.path.join(feature_root_dir, 'whole_cancer', target_feature, target_patch_size)

    whole_feature_list = []
    whole_type_list = []
    whole_center_list = []
    whole_race_list = []
    whole_sample_list = []
    whole_scanner_list = []
    df = df.drop_duplicates()

    if os.path.exists(os.path.join(data_save_dir, feature_type + '_embedding.npy')):
        whole_final_data = np.load(os.path.join(data_save_dir, feature_type + '_embedding.npy'))
        whole_final_df = pd.read_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'))
        whole_type_list = whole_final_df['subtype'].to_list()
        whole_center_list = whole_final_df['center'].to_list()
        whole_race_list = whole_final_df['race'].to_list()
        whole_sample_list = whole_final_df['sample'].to_list()
        if 'scanner' in whole_final_df.columns:
            whole_scanner_list = whole_final_df['scanner'].to_list()

    else:
        with tqdm(total = len(df)) as pbar:
            for i, target_df in df.iterrows():
                center = target_df['center']
                race = target_df['race']
                subtype = target_df['subtype']
                sample = target_df['sample']
                scanner = target_df['scanner']
                feature_file_name = target_df['feature_file_name']
                file_id = feature_file_name.split('.h5')[0]

                print(sample)
                whole_feature_rootdir = os.path.join(cancer_root, 'total_feature')
                target_data_path = os.path.join(whole_feature_rootdir, file_id + '.pt')
                if os.path.exists(target_data_path) is False:
                    print('No embedding data of sample: ' + sample)
                    continue
                else:
                    embedding = torch.load(target_data_path)
                    whole_cancer_embedding = torch.mean(embedding, axis = 0)
                    whole_feature_list.append(whole_cancer_embedding)
                    whole_type_list.append(subtype)
                    whole_center_list.append(center)
                    whole_race_list.append(race)
                    whole_scanner_list.append(scanner)
                    whole_sample_list.append(sample)
                pbar.update()

        whole_final_data = torch.stack(whole_feature_list, axis = 0).numpy()
        np.save(os.path.join(data_save_dir, feature_type + '_embedding.npy'), whole_final_data)

        whole_final_df = pd.DataFrame(zip(whole_type_list, whole_center_list, whole_race_list, whole_scanner_list, whole_sample_list), columns = ['subtype', 'center', 'race', 'scanner', 'sample'])
        whole_final_df.to_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'), index = False)
    
    # 1) Center prototypes
    prototype_dataframe, prototype = build_prototypes(
        whole_final_data, whole_final_df,
        group_cols=target_column,
        normalize_after_mean=False,
        normalize_X=True,
    )

    prototype_dataframe.to_csv(os.path.join(data_save_dir, f"{feature_type}_{target_column}_prototypes_meta.csv"), index=False)
    np.save(os.path.join(data_save_dir, f"{feature_type}_{target_column}_prototypes.npy"), np.stack([prototype[k] for k in prototype.keys()], axis=0))

    # Also save mapping of prototype order
    pd.DataFrame({"key": [str(k) for k in prototype.keys()]}).to_csv(os.path.join(data_save_dir, f"{feature_type}_{target_column}_prototypes_keys.csv"), index=False)

    return prototype

def prepare_testdata(feature_root_dir, df, save_dir, feature_type, target_patch_size, data_type):
    data_save_dir = os.path.join(save_dir, 'inference')
    os.makedirs(data_save_dir, exist_ok = True)

    if feature_type == 'UNI':
        target_feature = 'uni_v1'
    elif feature_type == 'UNI2':
        target_feature = 'uni_v2'
    elif feature_type == 'GigaPath':
        target_feature = 'gigapath'
    elif feature_type == 'CONCH':
        target_feature = 'conch_v1'
    else:
        target_feature = feature_type

    if 'stainnorm' in data_type:
        feature_type += '_stainnorm'
        target_feature += '_stainnorm'
    
    if 'stainnorm' in data_type:
        feature_type += '_stainnorm'
        target_feature += '_stainnorm'
    
    if 'canceronly' in data_type:
        cancer_root = os.path.join(feature_root_dir, 'cancer_only', target_feature, target_patch_size)

    else:
        cancer_root = os.path.join(feature_root_dir, 'whole_cancer', target_feature, target_patch_size)

    whole_feature_list = []
    whole_type_list = []
    whole_center_list = []
    whole_race_list = []
    whole_sample_list = []
    whole_scanner_list = []
    df = df.drop_duplicates()

    if os.path.exists(os.path.join(data_save_dir, feature_type + '_embedding.npy')):
        whole_final_data = np.load(os.path.join(data_save_dir, feature_type + '_embedding.npy'))
        whole_final_df = pd.read_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'))

    else:
        with tqdm(total = len(df)) as pbar:
            for i, target_df in df.iterrows():
                center = target_df['center']
                race = target_df['race']
                subtype = target_df['subtype']
                sample = target_df['sample']
                scanner = target_df['scanner']
                feature_file_name = target_df['feature_file_name']
                file_id = feature_file_name.split('.h5')[0]

                print(sample)
                whole_feature_rootdir = os.path.join(cancer_root, 'total_feature')
                target_data_path = os.path.join(whole_feature_rootdir, file_id + '.pt')
                if os.path.exists(target_data_path) is False:
                    print('No embedding data of sample: ' + sample)
                    continue
                else:
                    embedding = torch.load(target_data_path)
                    whole_feature_list.append(embedding)
                    whole_type_list.extend([subtype] * len(embedding))
                    whole_center_list.extend([center] * len(embedding))
                    whole_race_list.extend([race] * len(embedding))
                    whole_scanner_list.extend([scanner] * len(embedding))
                    whole_sample_list.extend([sample] * len(embedding))
                pbar.update()

        whole_final_data = torch.concat(whole_feature_list, axis = 0).numpy()
        whole_final_data = l2_normalize(whole_final_data, axis = 1)
        np.save(os.path.join(data_save_dir, feature_type + '_embedding.npy'), whole_final_data)

        whole_final_df = pd.DataFrame(zip(whole_type_list, whole_center_list, whole_race_list, whole_scanner_list, whole_sample_list), columns = ['subtype', 'center', 'race', 'scanner', 'sample'])
        whole_final_df.to_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'), index = False)

    return whole_final_df, whole_final_data

def compute_distances_to_prototypes(
    X_use: np.ndarray,
    meta: pd.DataFrame,
    protos: dict,
    group_cols,
    prefix: str,
    metric: str = "euclidean",
):
    """
    Compute distances from each sample embedding to each prototype.

    Returns:
      dist_wide: DataFrame with columns: group prototypes as separate columns
      dist_long: long-format distances with (sample, group_key, distance)
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    keys = list(protos.keys())
    P = np.stack([protos[k] for k in keys], axis=0)  # (G, D)

    if metric == "euclidean":
        # pairwise euclidean between X_use (N,D) and P (G,D)
        # ||x - p||^2 = ||x||^2 + ||p||^2 - 2 x·p
        x2 = np.sum(X_use**2, axis=1, keepdims=True)          # (N,1)
        p2 = np.sum(P**2, axis=1, keepdims=True).T            # (1,G)
        d2 = np.maximum(x2 + p2 - 2.0 * (X_use @ P.T), 0.0)   # (N,G)
        D = np.sqrt(d2)
    elif metric == "cosine":
        # cosine distance = 1 - cosine similarity
        # assumes X_use, P are L2-normalized if you want true cosine
        sim = X_use @ P.T
        D = 1.0 - sim
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    # build readable prototype names
    def key_to_name(k):
        if isinstance(k, tuple):
            return "__".join(map(str, k))
        return str(k)

    col_names = [f"{prefix}__{key_to_name(k)}" for k in keys]
    dist_wide = pd.DataFrame(D, columns=col_names)
    # attach identifiers
    dist_wide = pd.concat([meta.reset_index(drop=True), dist_wide], axis=1)

    # long format
    dist_long = dist_wide.melt(
        id_vars=list(meta.columns),
        value_vars=col_names,
        var_name=f"{prefix}_proto",
        value_name="distance",
    )
    return dist_wide, dist_long


def build_prototype_and_distance_calculation(feature_root_dir, metadata, target_column, target_center, pfm_list, save_dir, stainnorm = False, cancer_only = False, random_seed = 42):
    patch_size_dict = {'virchow': '224', 'UNI': '256', 'GigaPath': '256', 'virchow2': '224', 'UNI2': '256', 'CONCH': '512'}
    target_type_sample = pd.read_csv(metadata)
    target_type_sample = target_type_sample[target_type_sample['center'].isin(target_center)]
    meta_df = target_type_sample.drop_duplicates()

    save_dir = os.path.join(save_dir, 'prototype_distance_dataset')
    os.makedirs(save_dir, exist_ok=True)
    # ---- usage ----
    if os.path.exists(os.path.join(save_dir, 'prototype_df.csv')):
        prototype_df = pd.read_csv(os.path.join(save_dir, 'prototype_df.csv'))
        inference_df = pd.read_csv(os.path.join(save_dir, 'inference_df.csv'))
    else:
        prototype_df, inference_df, info = stratified_split_meta(meta_df, test_size=0.3, seed=random_seed, subtype_col="subtype", center_col="center")
        prototype_df.to_csv(os.path.join(save_dir, 'prototype_df.csv'), index = False)
        inference_df.to_csv(os.path.join(save_dir, 'inference_df.csv'), index = False)

    if stainnorm:
        if cancer_only:
            data_type = 'stainnorm_canceronly'
        else:
            data_type = 'stainnorm'
    else:
        if cancer_only:
            data_type = 'canceronly'
        else:
            data_type = 'original'


    print(data_type)
    data_type_save_dir = os.path.join(save_dir, data_type)
    os.makedirs(data_type_save_dir, exist_ok=True)

    for pfm in pfm_list:
        print(pfm)
        print('Build Prototype')
        prototype = build_prototype(feature_root_dir, prototype_df, target_column, data_type_save_dir, pfm, patch_size_dict[pfm], data_type)
        print('Done')
        
        print('Prepare testdata')
        inference_df_for_distance, inference_data = prepare_testdata(feature_root_dir, inference_df, data_type_save_dir, pfm, patch_size_dict[pfm], data_type)
        print('Done')

        data_distance_dir = os.path.join(data_type_save_dir, 'distance')
        print('Calculate_distance')
        dist_wide, dist_long = compute_distances_to_prototypes(
            inference_data, inference_df_for_distance, prototype, group_cols=target_column, prefix=target_column, metric="euclidean"
        )

        os.makedirs(os.path.join(data_type_save_dir, 'distance'), exist_ok = True)
        dist_wide.to_csv(os.path.join(data_distance_dir, f"{pfm}_dist_to_{target_column}_prototypes_wide.csv"), index=False)
        dist_long.to_csv(os.path.join(data_distance_dir, f"{pfm}_dist_to_{target_column}_prototypes_long.csv"), index=False)
        print("Done")

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--feature_rootdir", help = 'Processed feature root', type = str, required = True)
    parser.add_argument("--metadata", help = 'Path of the metadata', type = str, required = True)
    parser.add_argument("--save_dir", help = 'Directory to save the feature',type = str, required = True)
    parser.add_argument("--target_database", nargs = "+", default = 'TCGA', help = 'Database if you want to select all database data', type = str)
    parser.add_argument("--target_center", nargs = "+", default = ["MSKCC", "NCI Urologic Oncology Branch", "MD Anderson Cancer Center"], help = 'Center names you want to draw in data', type = str)
    parser.add_argument("--pfm_list", nargs = "+", default = ['virchow', 'virchow2', 'UNI', 'UNI2'], help = 'PFM list for comparison', type = str)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--stainnorm", default = False, help = 'Stainnorm data or not', type = bool)
    parser.add_argument("--cancer_only", default = False, help = 'Use cancer only data', type = bool)
    parser.add_argument("--random_seed", default = 42, help = 'Random seed for split', type = int)
    
    return parser.parse_args()

def main():
    Argument = Parser_main()
    build_prototype_and_distance_calculation(Argument.feature_rootdir, Argument.metadata, Argument.target_column, Argument.target_center, Argument.pfm_list, Argument.save_dir, Argument.stainnorm, Argument.cancer_only, Argument.random_seed)
        
if __name__ == "__main__":
    main()

        