import os
import umap
import torch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import re
import h5py
import pickle
from scipy.stats import ttest_ind
import math
import numpy as np
from collections import defaultdict
from tqdm import trange

import argparse

def p_to_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
torch.set_num_threads(10)

def bootstrap_bias(
    whole_final_data,
    whole_type_list,
    whole_center_list,
    whole_race_list,
    n_boot=10000,
    frac=0.7,
    base_seed=1234567,
    metric='NMI',
    method='KMeans'
):
    """
    부트스트랩으로 조합별( subtype × center × race ) 층화 샘플링을 하여
    매 반복마다 compare_clustering_fixed를 계산하고 결과를 수집합니다.

    Returns
    -------
    results : dict
        {
          'tumor_list': [mean_t ...],   # 길이 n_boot
          'center_list': [mean_c ...],  # 길이 n_boot
          'race_list': [mean_r ...],    # 길이 n_boot
          'indices_used': [np.ndarray, ...]  # 각 반복에서 사용된 전역 인덱스(옵션)
        }
    """
    rng = np.random.default_rng(base_seed)

    # numpy array로 고정
    whole_final_data = np.asarray(whole_final_data)
    type_arr   = np.asarray(whole_type_list)
    center_arr = np.asarray(whole_center_list)
    race_arr   = np.asarray(whole_race_list)

    assert len(whole_final_data) == len(type_arr) == len(center_arr) == len(race_arr), \
        "데이터와 라벨 길이가 일치해야 합니다."

    # 1) subtype-center-race 조합별 인덱스 그룹 만들기
    groups = defaultdict(list)
    for i, key in enumerate(zip(type_arr, center_arr, race_arr)):
        groups[key].append(i)

    # 각 그룹에서 뽑을 샘플 크기 미리 계산(ceil(0.7*n))
    take_sizes = {k: max(1, math.ceil(frac * len(idxs))) for k, idxs in groups.items()}

    tumor_list, center_list_res, race_list_res, scanner_list_res = [], [], [], []
    indices_used_per_boot = []

    for b in trange(n_boot, desc="Bootstrapping"):
        chosen_indices = []

        # 2) 그룹별로 ⌈0.7n⌉개씩 비복원 추출
        #    (그룹 크기가 매우 작아도 ceil 규칙에 따라 최소 1개는 선택)
        for key, idxs in groups.items():
            idxs = np.asarray(idxs)
            k = take_sizes[key]
            # k가 그룹 크기보다 클 일은 없음(ceil(0.7*n) ≤ n), 그래도 안전장치:
            k = min(k, len(idxs))
            sel = rng.choice(idxs, size=k, replace=False)
            chosen_indices.append(sel)

        chosen_indices = np.concatenate(chosen_indices)
        indices_used_per_boot.append(chosen_indices)

        # 3) 서브셋 생성
        X_sub = whole_final_data[chosen_indices]
        type_sub   = type_arr[chosen_indices].tolist()
        center_sub = center_arr[chosen_indices].tolist()
        race_sub   = race_arr[chosen_indices].tolist()


        mean_t, mean_c, mean_r = compare_clustering_fixed(
            X_sub, type_sub, center_sub, race_sub,
            random_state=int(rng.integers(0, 2**31-1)),
            metric=metric,
            method=method
        )

        tumor_list.append(float(mean_t))
        center_list_res.append(float(mean_c))
        race_list_res.append(float(mean_r))

    return {
        'tumor_list': tumor_list,
        'center_list': center_list_res,
        'race_list': race_list_res,
        'indices_used': indices_used_per_boot
    }

def bootstrap_bias_with_scanner(
    whole_final_data,
    whole_type_list,
    whole_center_list,
    whole_race_list,
    whole_scanner_list,
    n_boot=10000,
    frac=0.7,
    base_seed=1234567,
    metric='NMI',
    method='KMeans'
):
    rng = np.random.default_rng(base_seed)

    # numpy array로 고정
    whole_final_data = np.asarray(whole_final_data)
    type_arr   = np.asarray(whole_type_list)
    center_arr = np.asarray(whole_center_list)
    race_arr   = np.asarray(whole_race_list)
    scanner_arr = np.asarray(whole_scanner_list)

    assert len(whole_final_data) == len(type_arr) == len(center_arr) == len(race_arr) == len(scanner_arr), \
        "Length must be same"

    # 1) subtype-center-race 조합별 인덱스 그룹 만들기
    groups = defaultdict(list)
    for i, key in enumerate(zip(type_arr, center_arr, race_arr, scanner_arr)):
        groups[key].append(i)

    # 각 그룹에서 뽑을 샘플 크기 미리 계산(ceil(0.7*n))
    take_sizes = {k: max(1, math.ceil(frac * len(idxs))) for k, idxs in groups.items()}

    tumor_list, center_list_res, race_list_res, scanner_list_res = [], [], [], []
    indices_used_per_boot = []

    for b in trange(n_boot, desc="Bootstrapping"):
        chosen_indices = []

        # 2) 그룹별로 ⌈0.7n⌉개씩 비복원 추출
        #    (그룹 크기가 매우 작아도 ceil 규칙에 따라 최소 1개는 선택)
        for key, idxs in groups.items():
            idxs = np.asarray(idxs)
            k = take_sizes[key]
            # k가 그룹 크기보다 클 일은 없음(ceil(0.7*n) ≤ n), 그래도 안전장치:
            k = min(k, len(idxs))
            sel = rng.choice(idxs, size=k, replace=False)
            chosen_indices.append(sel)

        chosen_indices = np.concatenate(chosen_indices)
        indices_used_per_boot.append(chosen_indices)

        # 3) 서브셋 생성
        X_sub = whole_final_data[chosen_indices]
        type_sub   = type_arr[chosen_indices].tolist()
        center_sub = center_arr[chosen_indices].tolist()
        race_sub   = race_arr[chosen_indices].tolist()
        scanner_sub = scanner_arr[chosen_indices].tolist()


        mean_t, mean_c, mean_r, mean_s = compare_clustering_fixed_with_scanner(
            X_sub, type_sub, center_sub, race_sub, scanner_sub,
            random_state=int(rng.integers(0, 2**31-1)),
            metric=metric,
            method=method
        )

        tumor_list.append(float(mean_t))
        center_list_res.append(float(mean_c))
        race_list_res.append(float(mean_r))
        scanner_list_res.append(float(mean_s))

    return {
        'tumor_list': tumor_list,
        'center_list': center_list_res,
        'race_list': race_list_res,
        'scanner_list': scanner_list_res,
        'indices_used': indices_used_per_boot
    }

from collections import defaultdict
import numpy as np, math
from tqdm import trange

from collections import defaultdict
import numpy as np, math
from tqdm import trange

def bootstrap_bias_with_scanner_paired(
    whole_final_data,
    whole_type_list,
    whole_center_list,
    whole_race_list,
    whole_scanner_list,
    sample_list,                          # index별 sample_name
    n_boot=10000,
    frac=0.7,
    base_seed=1234567,
    metric='NMI',
    method='KMeans',
    *,
    base_center='GSH',                    # 페어의 "기준" center
    paired_center='GSH_histech',          # 페어의 "대상" center (추가 포함)
    base_scanner='Leica',                 # 선택사항: 기준 center의 scanner 라벨
    paired_scanner='Histech',             # 선택사항: 대상 center의 scanner 라벨
    enforce_scanner_labels=True           # True면 위 스캐너 라벨도 함께 필터링
):
    """
    로직:
      1) 부트스트랩 층화추출은 (center != paired_center) 모집단에서만 수행.
      2) 추출된 인덱스 중 (center == base_center)의 sample_name을 기준으로,
         (center == paired_center [그리고 scanner==paired_scanner, 옵션])의 같은 sample_name 인덱스를 '추가 포함'.
      3) 이후 compare_clustering_fixed_with_scanner를 그대로 호출.
    """
    rng = np.random.default_rng(base_seed)

    # numpy array 고정
    X          = np.asarray(whole_final_data)
    type_arr   = np.asarray(whole_type_list)
    center_arr = np.asarray(whole_center_list)
    race_arr   = np.asarray(whole_race_list)
    scan_arr   = np.asarray(whole_scanner_list)
    samp_arr   = np.asarray(sample_list)

    assert len(X) == len(type_arr) == len(center_arr) == len(race_arr) == len(scan_arr) == len(samp_arr), \
        "Length must be same"

    N = len(samp_arr)

    # --- (A) paired_center 인덱스 맵: sample_name -> [indices]
    paired_by_sample = defaultdict(list)
    for i in range(N):
        if center_arr[i] != paired_center:
            continue
        if enforce_scanner_labels and paired_scanner is not None:
            if scan_arr[i] != paired_scanner:
                continue
        paired_by_sample[samp_arr[i]].append(i)

    # --- (B) 부트스트랩 모집단: paired_center는 전부 제외
    base_mask = (center_arr != paired_center)
    base_idx = np.where(base_mask)[0]

    # --- (C) 층화 그룹핑: (type, center, race, scanner) 기준, but only on base_idx
    groups = defaultdict(list)
    for i in base_idx:
        key = (type_arr[i], center_arr[i], race_arr[i], scan_arr[i])
        groups[key].append(i)

    # 각 그룹에서 뽑을 크기 (ceil(frac*n), 최소 1)
    take_sizes = {k: max(1, math.ceil(frac * len(v))) for k, v in groups.items()}

    tumor_list, center_list_res, race_list_res, scanner_list_res = [], [], [], []
    indices_used_per_boot = []

    for _ in trange(n_boot, desc="Bootstrapping (paired center)"):
        chosen_base = []
        # (D) 그룹별 비복원 추출
        for key, idxs in groups.items():
            idxs = np.asarray(idxs)
            k = min(take_sizes[key], len(idxs))
            sel = rng.choice(idxs, size=k, replace=False)
            chosen_base.append(sel)

        chosen_base = np.concatenate(chosen_base)

        # (E) base_center에서 뽑힌 샘플들의 sample_name 수집
        mask_base_center = (center_arr[chosen_base] == base_center)
        selected_base_samples = np.unique(samp_arr[chosen_base][mask_base_center])

        # (F) 같은 sample_name의 paired_center 인덱스 추가
        add_list = []
        for sname in selected_base_samples:
            if sname in paired_by_sample:
                add_list.extend(paired_by_sample[sname])

        # (G) 최종 인덱스(중복 제거)
        if add_list:
            chosen_full = np.unique(np.concatenate([chosen_base, np.asarray(add_list, dtype=int)]))
        else:
            chosen_full = chosen_base

        indices_used_per_boot.append(chosen_full)

        # (H) 서브셋 생성 및 평가
        X_sub       = X[chosen_full]
        type_sub    = type_arr[chosen_full].tolist()
        center_sub  = center_arr[chosen_full].tolist()
        race_sub    = race_arr[chosen_full].tolist()
        scanner_sub = scan_arr[chosen_full].tolist()

        mean_t, mean_c, mean_r, mean_s = compare_clustering_fixed_with_scanner(
            X_sub, type_sub, center_sub, race_sub, scanner_sub,
            random_state=int(rng.integers(0, 2**31 - 1)),
            metric=metric,
            method=method
        )

        tumor_list.append(float(mean_t))
        center_list_res.append(float(mean_c))
        race_list_res.append(float(mean_r))
        scanner_list_res.append(float(mean_s))

    return {
        'tumor_list': tumor_list,
        'center_list': center_list_res,
        'race_list': race_list_res,
        'scanner_list': scanner_list_res,
        'indices_used': indices_used_per_boot
    }



def clean_id(id_str):
    # 공백 제거하고 숫자 앞의 0 제거
    id_str = id_str.replace("S ", "S")
    parts = id_str.split("-")
    prefix = parts[0]
    number = str(int(parts[1]))  # 앞자리 0 제거
    return f"{prefix}-{number}"

def extract_significant_digits(sample_name):
    match = re.search(r'-0*([1-9]\d*)', sample_name)
    if match:
        return match.group(1)
    return None

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def compare_clustering_fixed(
    feature_list,
    type_list,
    center_list,
    race_list,
    n_subtype_clusters = 3,
    n_center_clusters = 3,
    n_race_clusters = 3, 
    random_state=None,
    method='KMeans',
    metric='NMI'
):
    """
    한 번만 클러스터링하고 각 라벨과 일치도(NMI or ARI)를 계산합니다.
    """

    feature_array = feature_list  # (N, d)

    type_arr = np.asarray(type_list)
    center_arr = np.asarray(center_list)
    race_arr = np.asarray(race_list)

    # Clustering
    if method == 'GMM':
        subtype_model = GaussianMixture(n_components=n_subtype_clusters, covariance_type='full', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = GaussianMixture(n_components=n_center_clusters, covariance_type='full', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = GaussianMixture(n_components=n_race_clusters, covariance_type='full', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)

    elif method == 'Spectral':
        subtype_model = SpectralClustering(n_clusters=n_subtype_clusters, affinity='nearest_neighbors', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = SpectralClustering(n_clusters=n_center_clusters, affinity='nearest_neighbors', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = SpectralClustering(n_clusters=n_race_clusters, affinity='nearest_neighbors', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)

    elif method == 'Agglomerative':
        subtype_model = AgglomerativeClustering(n_clusters=n_subtype_clusters)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = AgglomerativeClustering(n_clusters=n_center_clusters)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = AgglomerativeClustering(n_clusters=n_race_clusters)
        race_pred_labels = race_model.fit_predict(feature_array)

    elif method == 'KMeans':
        subtype_model = KMeans(n_clusters=n_subtype_clusters, random_state=random_state, n_init='auto')
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = KMeans(n_clusters=n_center_clusters, random_state=random_state, n_init='auto')
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = KMeans(n_clusters=n_race_clusters, random_state=random_state, n_init='auto')
        race_pred_labels = race_model.fit_predict(feature_array)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Score with each label
    if metric == 'NMI':
        score_fn = normalized_mutual_info_score
    elif metric == 'ARI':
        score_fn = adjusted_rand_score
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    score_type = score_fn(type_arr, subtype_pred_labels)
    score_center = score_fn(center_arr, center_pred_labels)
    score_race = score_fn(race_arr, race_pred_labels)

    return score_type, score_center, score_race

def compare_clustering_fixed_with_scanner(
    feature_list,
    type_list,
    center_list,
    race_list,
    scanner_list,
    n_subtype_clusters = 3,
    n_center_clusters = 3,
    n_race_clusters = 3, 
    n_scanner_clusters = 2,
    random_state=None,
    method='KMeans',
    metric='NMI'
):
    """
    한 번만 클러스터링하고 각 라벨과 일치도(NMI or ARI)를 계산합니다.
    """

    feature_array = feature_list  # (N, d)

    type_arr = np.asarray(type_list)
    center_arr = np.asarray(center_list)
    race_arr = np.asarray(race_list)
    scanner_arr = np.asarray(scanner_list)

    # Clustering
    if method == 'GMM':
        subtype_model = GaussianMixture(n_components=n_subtype_clusters, covariance_type='full', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = GaussianMixture(n_components=n_center_clusters, covariance_type='full', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = GaussianMixture(n_components=n_race_clusters, covariance_type='full', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = GaussianMixture(n_components=n_scanner_clusters, covariance_type='full', random_state=random_state)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'Spectral':
        subtype_model = SpectralClustering(n_clusters=n_subtype_clusters, affinity='nearest_neighbors', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = SpectralClustering(n_clusters=n_center_clusters, affinity='nearest_neighbors', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = SpectralClustering(n_clusters=n_race_clusters, affinity='nearest_neighbors', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = SpectralClustering(n_clusters=n_scanner_clusters, affinity='nearest_neighbors', random_state=random_state)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'Agglomerative':
        subtype_model = AgglomerativeClustering(n_clusters=n_subtype_clusters)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = AgglomerativeClustering(n_clusters=n_center_clusters)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = AgglomerativeClustering(n_clusters=n_race_clusters)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = AgglomerativeClustering(n_clusters=n_scanner_clusters)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'KMeans':
        subtype_model = KMeans(n_clusters=n_subtype_clusters, random_state=random_state, n_init='auto')
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = KMeans(n_clusters=n_center_clusters, random_state=random_state, n_init='auto')
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = KMeans(n_clusters=n_race_clusters, random_state=random_state, n_init='auto')
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = KMeans(n_clusters=n_scanner_clusters, random_state=random_state, n_init='auto')
        scanner_pred_labels = scanner_model.fit_predict(feature_array)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Score with each label
    if metric == 'NMI':
        score_fn = normalized_mutual_info_score
    elif metric == 'ARI':
        score_fn = adjusted_rand_score
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    score_type = score_fn(type_arr, subtype_pred_labels)
    score_center = score_fn(center_arr, center_pred_labels)
    score_race = score_fn(race_arr, race_pred_labels)
    score_scanner = score_fn(scanner_arr, scanner_pred_labels)

    return score_type, score_center, score_race, score_scanner

def compare_clustering_fixed_with_scanner(
    feature_list,
    type_list,
    center_list,
    race_list,
    scanner_list,
    n_subtype_clusters = 3,
    n_center_clusters = 3,
    n_race_clusters = 3, 
    n_scanner_clusters = 2,
    random_state=None,
    method='KMeans',
    metric='NMI'
):
    """
    한 번만 클러스터링하고 각 라벨과 일치도(NMI or ARI)를 계산합니다.
    """

    feature_array = feature_list  # (N, d)

    type_arr = np.asarray(type_list)
    center_arr = np.asarray(center_list)
    race_arr = np.asarray(race_list)
    scanner_arr = np.asarray(scanner_list)

    # Clustering
    if method == 'GMM':
        subtype_model = GaussianMixture(n_components=n_subtype_clusters, covariance_type='full', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = GaussianMixture(n_components=n_center_clusters, covariance_type='full', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = GaussianMixture(n_components=n_race_clusters, covariance_type='full', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = GaussianMixture(n_components=n_scanner_clusters, covariance_type='full', random_state=random_state)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'Spectral':
        subtype_model = SpectralClustering(n_clusters=n_subtype_clusters, affinity='nearest_neighbors', random_state=random_state)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = SpectralClustering(n_clusters=n_center_clusters, affinity='nearest_neighbors', random_state=random_state)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = SpectralClustering(n_clusters=n_race_clusters, affinity='nearest_neighbors', random_state=random_state)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = SpectralClustering(n_clusters=n_scanner_clusters, affinity='nearest_neighbors', random_state=random_state)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'Agglomerative':
        subtype_model = AgglomerativeClustering(n_clusters=n_subtype_clusters)
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = AgglomerativeClustering(n_clusters=n_center_clusters)
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = AgglomerativeClustering(n_clusters=n_race_clusters)
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = AgglomerativeClustering(n_clusters=n_scanner_clusters)
        scanner_pred_labels = scanner_model.fit_predict(feature_array)

    elif method == 'KMeans':
        subtype_model = KMeans(n_clusters=n_subtype_clusters, random_state=random_state, n_init='auto')
        subtype_pred_labels = subtype_model.fit_predict(feature_array)
        center_model = KMeans(n_clusters=n_center_clusters, random_state=random_state, n_init='auto')
        center_pred_labels = center_model.fit_predict(feature_array)
        race_model = KMeans(n_clusters=n_race_clusters, random_state=random_state, n_init='auto')
        race_pred_labels = race_model.fit_predict(feature_array)
        scanner_model = KMeans(n_clusters=n_scanner_clusters, random_state=random_state, n_init='auto')
        scanner_pred_labels = scanner_model.fit_predict(feature_array)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Score with each label
    if metric == 'NMI':
        score_fn = normalized_mutual_info_score
    elif metric == 'ARI':
        score_fn = adjusted_rand_score
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    score_type = score_fn(type_arr, subtype_pred_labels)
    score_center = score_fn(center_arr, center_pred_labels)
    score_race = score_fn(race_arr, race_pred_labels)
    score_scanner = score_fn(scanner_arr, scanner_pred_labels)

    return score_type, score_center, score_race, score_scanner


def draw_figure(df, output_dir, columns, name):
    # Calculate accuracy per type and prediction column
    types = df['type'].unique()
    accuracy = {}

    for col in columns:
        accuracy[col] = {}
        for t in types:
            correct_predictions = df[(df['type'] == t) & (df[col] == t)].shape[0]
            total_predictions = df[df['type'] == t].shape[0]
            accuracy[col][t] = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Transform accuracy to a dataframe for plotting
    accuracy_df = pd.DataFrame(accuracy).reset_index().rename(columns={'index': 'Type'})

    # Plot accuracy barplot
    plt.figure(figsize=(10, 6))
    accuracy_long = pd.melt(accuracy_df, id_vars='Type', var_name='Column', value_name='Accuracy')
    sns.barplot(data=accuracy_long, x='Column', y='Accuracy', hue='Type')
    plt.title('Accuracy by Type and Column')
    plt.ylabel('Accuracy')
    plt.xlabel('Prediction Columns')
    plt.legend(title='Type')
    plt.tight_layout()

    # Save the barplot
    os.makedirs(output_dir, exist_ok=True)
    barplot_path = os.path.join(output_dir, f'accuracy_barplot_{name}.png')
    plt.savefig(barplot_path)
    plt.close()

    # Confusion matrix calculation and plotting for each prediction column
    for col in columns:
        cm = pd.crosstab(df['type'], df[col], rownames=['Actual'], colnames=['Predicted'], normalize='index')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Confusion Matrix for {col}')
        plt.tight_layout()
        
        # Save the confusion matrix plot
        cm_path = os.path.join(output_dir, f'confusion_matrix_{col}_{name}.png')
        plt.savefig(cm_path)
        plt.close()

def bootstrapping_bias_calculation(df, save_dir, feature_type, target_patch_size, data_type, database):
    data_save_dir = os.path.join(save_dir, 'processed_data')
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
    
    if 'TCGA' in database:
        if 'canceronly' in data_type:
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/cancer_only/{target_feature}/{target_patch_size}/"
        else:
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/whole_cancer/{target_feature}/{target_patch_size}/"

    elif 'internal' in database:
        if 'canceronly' in data_type:
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Inference/cancer_only/"
        else:
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Inference/whole_cancer/"

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
        whole_type_list = whole_final_df['Subtype'].to_list()
        whole_center_list = whole_final_df['Center'].to_list()
        whole_race_list = whole_final_df['Race'].to_list()
        whole_sample_list = whole_final_df['Sample'].to_list()
        if 'Scanner' in whole_final_df.columns:
            whole_scanner_list = whole_final_df['Scanner'].to_list()

    else:
        with tqdm(total = len(df)) as pbar:
            for i, target_df in df.iterrows():
                center = target_df['center']
                race = target_df['race']
                subtype = target_df['subtype']
                sample = target_df['sample']
                scanner = target_df['scanner']
                print(sample)
                
                if 'TCGA' in database:
                    whole_feature_rootdir= os.path.join(whole_cancer_root, subtype, 'total_feature')
                elif 'internal' in database:
                    whole_feature_rootdir= os.path.join(whole_cancer_root, center, target_feature, target_patch_size, subtype, 'total_feature')
                whole_data_list = os.listdir(whole_feature_rootdir)
                target_data_list = [d for d in whole_data_list if sample in d]
                if len(target_data_list) != 0:
                    target_data = target_data_list[0]
                    embedding = torch.load(os.path.join(whole_feature_rootdir, target_data))
                    whole_cancer_embedding = torch.mean(embedding, axis = 0)
                    whole_feature_list.append(whole_cancer_embedding)
                    whole_type_list.append(subtype)
                    whole_center_list.append(center)
                    whole_race_list.append(race)
                    whole_scanner_list.append(scanner)
                    whole_sample_list.append(sample)
                else:
                    print(sample)
                    continue
                
                pbar.update()

        whole_final_data = torch.stack(whole_feature_list, axis = 0).numpy()
        np.save(os.path.join(data_save_dir, feature_type + '_embedding.npy'), whole_final_data)

        whole_final_df = pd.DataFrame(zip(whole_type_list, whole_center_list, whole_race_list, whole_scanner_list, whole_sample_list), columns = ['Subtype', 'Center', 'Race', 'Scanner', 'Sample'])
        whole_final_df.to_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'), index = False)

    if 'scanner' in data_type:
        results = bootstrap_bias_with_scanner_paired(
            whole_final_data,
            whole_type_list,
            whole_center_list,
            whole_race_list,
            whole_scanner_list,
            whole_sample_list,
            n_boot=1000,
            frac=0.7,
            base_seed=20250817,   
            metric='NMI',
            method='KMeans'
            )
        t_list = results['tumor_list']
        c_list = results['center_list']
        r_list = results['race_list']
        s_list = results['scanner_list']
        return t_list, c_list, r_list, s_list
    
    else:
        results = bootstrap_bias(
            whole_final_data,
            whole_type_list,
            whole_center_list,
            whole_race_list,
            n_boot=1000,
            frac=0.7,
            base_seed=20250817,
            metric='NMI',
            method='KMeans'
            )
    
        t_list = results['tumor_list']
        c_list = results['center_list']
        r_list = results['race_list']
        
        return t_list, c_list, r_list 


def calculate_NMI_bootstrap(metadata, target_center, pfm_list, save_dir, database):
    Kidney_df = pd.read_csv(metadata)
    target_data_list = ['original', 'stainnorm', 'canceronly', 'stainnorm_canceronly']
    patch_size_dict = {'virchow': '224', 'UNI': '256', 'GigaPath': '256', 'virchow2': '224', 'UNI2': '256', 'CONCH': '512'}
    database_dir = os.path.join(save_dir, database)
    target_data_bias_dict = {}
    for target_data in target_data_list:
        target_dir = os.path.join(database_dir, target_data)
        os.makedirs(target_dir, exist_ok=True)

        target_type = ['KIRC', 'KICH', 'KIRP']
        bias_dict = {}
        for pfm in pfm_list:
            target_patch_size = patch_size_dict[pfm]
            print(pfm)
            target_type_sample = Kidney_df[Kidney_df['subtype'].isin(target_type)]
            target_type_sample = target_type_sample[target_type_sample['center'].isin(target_center)]
            target_type_sample = target_type_sample[target_type_sample['race'] != 'not reported']
            target_type_sample = target_type_sample[target_type_sample['race'] != 'american indian or alaska native']
            
            bias_dict[pfm] = {}
            if 'scanner' in database:
                t_list, c_list, r_list, s_list = bootstrapping_bias_calculation(target_type_sample, target_dir, pfm, target_patch_size, target_data, database)
                bias_dict[pfm]['scanner'] = s_list
            else:
                t_list, c_list, r_list = bootstrapping_bias_calculation(target_type_sample, target_dir, pfm, target_patch_size, target_data, database)
            
            bias_dict[pfm]['subtype'] = t_list
            bias_dict[pfm]['center'] = c_list
            bias_dict[pfm]['race'] = r_list

        bias_data_dir = os.path.join(target_dir, 'bootstrap_result')
        os.makedirs(bias_data_dir, exist_ok = True)
        with open(os.path.join(bias_data_dir, 'UMAP_clustering_comparison.pickle'), 'wb') as f:
            pickle.dump(bias_dict, f, pickle.HIGHEST_PROTOCOL)
        target_data_bias_dict[target_data] = bias_dict
    
    conditions = ["original", "stainnorm", "canceronly", "stainnorm_canceronly"]
    condition_colors = {
        "original": "#f3d0d4",
        "stainnorm": "#ea9ab1",
        "canceronly": "#cb688c",
        "stainnorm_canceronly": "#684060",
    }

    metrics = [
        ("subtype", "Subtype NMI"),
        ("center", "Center NMI"),
        ("race", "Race NMI"),
    ]

    pfms_order = ["virchow", "virchow2", "UNI", "UNI2", "GigaPath", "CONCH"]
    available_pfms = list(next(iter(target_data_bias_dict.values())).keys())
    pfms = [p for p in pfms_order if p in available_pfms]
    n_pfms = len(pfms)
    n_cond = len(conditions)

    # 비교할 pair (항상 original vs X)
    # offset_idx는 같은 PFM 내에서 세 쌍 별표 높이를 조금씩 다르게 주기 위한 index
    star_pairs = [
        ("stainnorm", 0),
        ("canceronly", 1),
        ("stainnorm_canceronly", 2),
    ]
    for metric_key, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(1.8 * n_pfms, 5))

        x = np.arange(n_pfms)
        bar_width = 0.18
        offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * bar_width

        means = {}
        ci_lowers = {}
        ci_uppers = {}
        max_upper_for_ylim = -np.inf

        # 각 condition, PFM에 대해 mean + 95% CI 계산
        for cond_idx, cond in enumerate(conditions):
            means[cond] = []
            ci_lowers[cond] = []
            ci_uppers[cond] = []

            for pfm in pfms:
                data = np.array(target_data_bias_dict[cond][pfm][metric_key])
                mean_val = data.mean()
                lower, upper = np.percentile(data, [2.5, 97.5])  # 95% CI

                means[cond].append(mean_val)
                ci_lowers[cond].append(lower)
                ci_uppers[cond].append(upper)

                if upper > max_upper_for_ylim:
                    max_upper_for_ylim = upper

            means[cond] = np.array(means[cond])
            ci_lowers[cond] = np.array(ci_lowers[cond])
            ci_uppers[cond] = np.array(ci_uppers[cond])

            yerr = np.vstack([
                means[cond] - ci_lowers[cond],
                ci_uppers[cond] - means[cond],
            ])

            ax.bar(
                x + offsets[cond_idx],
                means[cond],
                width=bar_width,
                yerr=yerr,
                capsize=3,
                color=condition_colors[cond],
                edgecolor="black",
                label=cond,
            )

        # y축 범위 (CI 위로 약간 여유)
        ax.set_ylim(0, max_upper_for_ylim * 1.4)

        ax.set_xticks(x)
        ax.set_xticklabels(pfms, rotation=45, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_xlabel("PFM")
        ax.set_title(f"{metric_label} (bootstrap mean ± 95% CI)")
        ax.legend(title="Condition")

        # --------------------------------------
        # original vs (stainnorm / canceronly / stainnorm_canceronly) t-test + 별표
        # --------------------------------------
        for pfm_idx, pfm in enumerate(pfms):
            data_orig = np.array(target_data_bias_dict["original"][pfm][metric_key])
            upper_orig = np.percentile(data_orig, 97.5)

            x_orig = x[pfm_idx] + offsets[conditions.index("original")]

            for other_cond, offset_idx in star_pairs:
                data_other = np.array(target_data_bias_dict[other_cond][pfm][metric_key])

                # Welch's t-test
                t_stat, p_val = ttest_ind(data_orig, data_other, equal_var=False)
                star = p_to_star(p_val)
                if star == "":
                    continue  # 유의하지 않으면 skip

                upper_other = np.percentile(data_other, 97.5)
                x_other = x[pfm_idx] + offsets[conditions.index(other_cond)]

                # 두 bar를 덮는 괄호를 그릴 위치
                y_base = max(upper_orig, upper_other)
                # 같은 PFM 안에서 비교쌍마다 높이 조금씩 다르게 (0, 1, 2)
                # 계수 1.08, 1.16, 1.24 정도로 띄움
                height_factor = 1.08 + 0.08 * offset_idx
                y_line = y_base * height_factor

                # 괄호 ( ┐ ┌ 모양)
                ax.plot(
                    [x_orig, x_orig, x_other, x_other],
                    [y_line * 0.98, y_line, y_line, y_line * 0.98],
                    color="black",
                    linewidth=1,
                )

                # 괄호 중앙에 별 표시
                x_mid = (x_orig + x_other) / 2.0
                ax.text(
                    x_mid,
                    y_line * 1.01,
                    star,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.tight_layout()

        # 저장만 (show는 호출하지 않음)
        png_path = os.path.join(database_dir, f"{metric_key}_bias_barplot.png")
        svg_path = os.path.join(database_dir, f"{metric_key}_bias_barplot.svg")
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)

        plt.close(fig)

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--metadata", default = "/mnt/disk1/Kidney/Submission_dir/metadata.csv", help = 'Path of the metadata', type = str)
    parser.add_argument("--target_center", nargs = "+", default = ['BORAMAE', 'KHMC', 'GSH', 'SNUBH'], help = 'NCI Urologic Oncology Branch, MSKCC, MD Anderson Cancer Center or BORAMAE, KHMC, GSH, GSH_histech, SNUBH', type = str)
    parser.add_argument("--pfm_list", nargs = "+", default = ['virchow', 'virchow2' , 'UNI', 'UNI2', 'GigaPath', 'CONCH'], help = 'TCGA or Internal center, maybe another database can be used if you have', type = str)
    parser.add_argument("--save_dir", default = '/mnt/disk1/Kidney/Submission_dir/Figure3/final_data/', help = 'Directory to save the feature',type = str)
    parser.add_argument("--database", default = 'internal', help = 'TCGA or internal', type = str)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    calculate_NMI_bootstrap(Argument.metadata, Argument.target_center,  Argument.pfm_list, Argument.save_dir, Argument.database)
        
if __name__ == "__main__":
    main()
