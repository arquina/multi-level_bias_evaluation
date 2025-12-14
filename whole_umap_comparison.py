import os
import umap
import torch
from matplotlib import pyplot as plt
import pandas as pd
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import re
import pickle
import argparse

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

def draw_umap(umap_data, umap_dir, type_list, sample_name_list, name, feature_name, type_dict):
    umap_figure_dir = os.path.join(umap_dir, 'figure')
    os.makedirs(umap_figure_dir, exist_ok=True)
    umap_data_dir = os.path.join(umap_dir, 'UMAP_data')
    os.makedirs(umap_data_dir, exist_ok=True)
    
    # Convert type_list (numbers) to categorical names using the type_dict
    df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2']) 
    df['Sample_Name'] = sample_name_list    
    df['Type'] = type_list
    df.to_csv(os.path.join(umap_data_dir, f'umap_data_{name}.csv'), index=False)

    hue_order = [t for t in list(type_dict.keys()) if t in df['Type'].tolist()]

    palette = {'KIRC': '#DD6274',
               'KIRP': '#F6AE62',
            'KICH': '#786DB0',
            'normal': '#666A73',
            'white': '#E399C3',
            'asian':   '#1E1854',
            'black or african american': '#508AE2',
            'MSKCC': '#33CC99',
            'MD Anderson Cancer Center': '#0D90BF',
            'NCI Urologic Oncology Branch': '#595959',
            'BORAMAE': '#CC00CC',
            'KHMC': '#E5DA0B',
            'SNUBH': '#C7C7DC',
            'GSH': '#339933',
            'GSH_histech': "#14FF33",
            'Histech': "#DB19C4",
            'Leica': "#1982EA"}

    # Calculate new axis limits
    center_x = (df['UMAP1'].max() + df['UMAP1'].min()) / 2
    center_y = (df['UMAP2'].max() + df['UMAP2'].min()) / 2

    half_range_x = (df['UMAP1'].max() - center_x) * 1.5
    half_range_y = (df['UMAP2'].max() - center_y) * 1.5

    new_xlim = (center_x - half_range_x, center_x + half_range_x)
    new_ylim = (center_y - half_range_y, center_y + half_range_y)

    # Plot using seaborn
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Type', data=df, palette=palette, s=100, edgecolor=None, hue_order=hue_order)
    
    # Set plot titles and labels
    plt.title(f'UMAP of Kidney Cancer')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Set the new x and y limits
    plt.xlim(new_xlim)
    plt.ylim(new_ylim)
    
    # Save the plot
    svg_path = os.path.join(umap_figure_dir, f'{feature_name}_UMAP_of_{name}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)

    pdf_path = os.path.join(umap_figure_dir, f'{feature_name}_UMAP_of_{name}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)

    png_path = os.path.join(umap_figure_dir, f'{feature_name}_UMAP_of_{name}.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

def UMAP_drawing_and_bias_calculation(df, umap_figure_dir, feature_type, target_patch_size, data_type, stainnorm = False, cancer_annotation = False, multi_scanner = False, with_normal = False):
    data_save_dir = os.path.join(umap_figure_dir, 'processed_data')
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

    if stainnorm:
        feature_type += '_stainnorm'
        target_feature += '_stainnorm'
    
    if multi_scanner:
        feature_type += '_multi_scanner'
    
    if with_normal:
        feature_type += '_with_normal'
    
    
    if 'TCGA' in data_type:
        if cancer_annotation:
            if 'normal' in data_type:
                normal_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/non_cancer/{target_feature}/{target_patch_size}/"
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/cancer_only/{target_feature}/{target_patch_size}/"
        else:
            whole_cancer_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Prototype/whole_cancer/{target_feature}/{target_patch_size}/"

    elif 'internal' in data_type:
        if cancer_annotation:
            if 'normal' in data_type:
                normal_root = f"/mnt/disk1/Kidney/Submission_dir/dataset/20x/Inference/non_cancer/"
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
                if 'TCGA' in data_type:
                    whole_feature_rootdir= os.path.join(whole_cancer_root, subtype, 'total_feature')
                    if 'normal' in data_type:
                        whole_normal_feature_rootdir = os.path.join(normal_root, subtype, 'total_feature')
                elif 'internal' in data_type:
                    whole_feature_rootdir= os.path.join(whole_cancer_root, center, target_feature, target_patch_size, subtype, 'total_feature')
                    if 'normal' in data_type:
                        whole_normal_feature_rootdir = os.path.join(normal_root, subtype, 'total_feature')
                
                whole_data_list = os.listdir(whole_feature_rootdir)
                target_data_list = [d for d in whole_data_list if sample in d]
                if 'normal' in data_type:
                    whole_normal_data_list = os.listdir(whole_normal_feature_rootdir)
                    target_normal_data_list = [d for d in whole_normal_data_list if sample in d]

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
                if 'normal' in data_type:
                    if len(target_normal_data_list) != 0:
                        target_normal_data = target_normal_data_list[0]
                        normal_embedding = torch.load(os.path.join(whole_normal_feature_rootdir, target_normal_data))
                        whole_normal_embedding = torch.mean(normal_embedding, axis = 0)
                        whole_feature_list.append(whole_normal_embedding)
                        whole_type_list.append('normal')
                        whole_center_list.append(center)
                        whole_race_list.append(race)
                        whole_scanner_list.append(scanner)
                        whole_sample_list.append(sample)
                
                pbar.update()

        whole_final_data = torch.stack(whole_feature_list, axis = 0).numpy()
        np.save(os.path.join(data_save_dir, feature_type + '_embedding.npy'), whole_final_data)

        whole_final_df = pd.DataFrame(zip(whole_type_list, whole_center_list, whole_race_list, whole_scanner_list, whole_sample_list), columns = ['Subtype', 'Center', 'Race', 'Scanner', 'Sample'])
        whole_final_df.to_csv(os.path.join(data_save_dir, feature_type + '_metadata.csv'), index = False)
    
    whole_reducer = umap.UMAP(n_components=2, random_state = 12345)
    whole_umap_data = whole_reducer.fit_transform(whole_final_data)
    
    type_dict = {}
    for i, t in enumerate(list(set(whole_type_list))):
        type_dict[t] = i
    
    race_dict = {}
    for i, c in enumerate(list(set(whole_race_list))):
        race_dict[c] = i
    
    center_dict = {}
    for i, c in enumerate(list(set(whole_center_list))):
        center_dict[c] = i
    
    if multi_scanner:
        scanner_dict = {}
        for i, c in enumerate(list(set(whole_scanner_list))):
            scanner_dict[c] = i
    
    draw_umap(whole_umap_data, umap_figure_dir, whole_type_list, whole_sample_list, f'subtype_{feature_type}_whole', feature_type, type_dict)
    draw_umap(whole_umap_data, umap_figure_dir, whole_center_list, whole_sample_list, f'center_{feature_type}_whole', feature_type, center_dict)
    draw_umap(whole_umap_data, umap_figure_dir, whole_race_list, whole_sample_list, f'race_{feature_type}_whole', feature_type, race_dict)

    if multi_scanner:
        draw_umap(whole_umap_data, umap_figure_dir, whole_scanner_list, whole_sample_list, f'scanner_{feature_type}_whole', feature_type, scanner_dict)
    
    if multi_scanner:
        mean_t, mean_c, mean_r, mean_s = compare_clustering_fixed_with_scanner(
                whole_final_data,
                whole_type_list,
                whole_center_list,
                whole_race_list,
                whole_scanner_list,
                len(list(set(whole_type_list))),
                len(list(set(whole_center_list))),
                len(list(set(whole_race_list))),
                len(list(set(whole_scanner_list))),
                random_state=1234567,
                metric='NMI',
                method='KMeans'
            )
        return mean_t, mean_c, mean_r, mean_s
    else:
        mean_t, mean_c, mean_r = compare_clustering_fixed(
                whole_final_data,
                whole_type_list,
                whole_center_list,
                whole_race_list,
                len(list(set(whole_type_list))),
                len(list(set(whole_center_list))),
                len(list(set(whole_race_list))),
                random_state=1234567,
                metric='NMI',
                method='KMeans'
            )
        return mean_t, mean_c, mean_r
    
def draw_umap_and_calculate_NMI(metadata, target_center, pfm_list, save_dir, target_data, stainnorm,cancer_annotation = False, multi_scanner = False, with_normal = False):
    Kidney_df = pd.read_csv(metadata)
    patch_size_dict = {'virchow': '224', 'UNI': '256', 'GigaPath': '256', 'virchow2': '224', 'UNI2': '256', 'CONCH': '512'}

    if stainnorm:
        target_data += '_stainnorm'
    if multi_scanner:
        target_data += '_multi_scanner'
    if with_normal:
        target_data += '_with_normal'

    target_dir = os.path.join(save_dir, target_data)
    os.makedirs(target_dir, exist_ok=True)
    umap_figure_dir = os.path.join(target_dir, 'UMAP')
    os.makedirs(umap_figure_dir, exist_ok=True)
    bias_figure_dir = os.path.join(target_dir, 'bias')
    os.makedirs(bias_figure_dir, exist_ok = True)

    target_type = ['KIRC', 'KICH', 'KIRP']
    bias_dict = {}
    for pfm in pfm_list:
        target_patch_size = patch_size_dict[pfm]
        print(pfm)
        target_type_sample = Kidney_df[Kidney_df['subtype'].isin(target_type)]
        target_type_sample = target_type_sample[target_type_sample['center'].isin(target_center)]
        target_type_sample = target_type_sample[target_type_sample['race'] != 'not reported']
        target_type_sample = target_type_sample[target_type_sample['race'] != 'american indian or alaska native']
        
        if multi_scanner:
            mean_t, mean_c, mean_r, mean_s = UMAP_drawing_and_bias_calculation(target_type_sample, umap_figure_dir, pfm, target_patch_size, target_data, stainnorm,cancer_annotation, multi_scanner, with_normal)
            bias_dict[pfm] = [mean_t, mean_c, mean_r, mean_s]
        else:
            mean_t, mean_c, mean_r = UMAP_drawing_and_bias_calculation(target_type_sample, umap_figure_dir, pfm, target_patch_size, target_data, stainnorm,cancer_annotation, multi_scanner, with_normal)
            bias_dict[pfm] = [mean_t, mean_c, mean_r]

    if multi_scanner:
        df = pd.DataFrame.from_dict(bias_dict, orient="index", columns=["Subtype", "Center", "Race", "Scanner"])

        csv_path = os.path.join(bias_figure_dir, "NMI_value.csv")
        df.to_csv(csv_path)

        keys = df.index.tolist()
        values = df.values
        colors = ["#a4c6d8", "#62778a", "#9e9fbc", "#c0579c"]
        fig_names = ["Subtype", "Center", "Race", 'Scanner']

        for i in range(4):
            plt.figure()
            plt.bar(keys, values[:, i], color=colors[i])
            plt.ylim(0, 1)
            plt.xlabel("PFM")
            plt.ylabel(f"{fig_names[i]}")
            plt.xticks(rotation=45)

            png_path = os.path.join(bias_figure_dir, f"{fig_names[i]}.png")
            svg_path = os.path.join(bias_figure_dir, f"{fig_names[i]}.svg")
            plt.tight_layout()
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path)
            plt.close()
    else:
        df = pd.DataFrame.from_dict(bias_dict, orient="index", columns=["Subtype", "Center", "Race"])

        csv_path = os.path.join(bias_figure_dir, "NMI_value.csv")
        df.to_csv(csv_path)

        keys = df.index.tolist()
        values = df.values
        colors = ["#a4c6d8", "#62778a", "#9e9fbc"]

        fig_names = ["Subtype", "Center", "Race"]
        for i in range(3):
            plt.figure()
            plt.bar(keys, values[:, i], color=colors[i])
            plt.ylim(0, 1)
            plt.xlabel("PFM")
            plt.ylabel(f"{fig_names[i]}")
            plt.xticks(rotation=45)

            png_path = os.path.join(bias_figure_dir, f"{fig_names[i]}.png")
            svg_path = os.path.join(bias_figure_dir, f"{fig_names[i]}.svg")
            plt.tight_layout()
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path)
            plt.close()



def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--metadata", default = "/mnt/disk1/Kidney/Submission_dir/metadata.csv", help = 'Path of the metadata', type = str)
    parser.add_argument("--target_center", nargs = "+", default = ['BORAMAE', 'KHMC', 'GSH', 'GSH_histech', 'SNUBH'], help = 'Center names you want to draw in data', type = str)
    parser.add_argument("--pfm_list", nargs = "+", default = ['virchow', 'virchow2' , 'UNI', 'UNI2', 'GigaPath', 'CONCH'], help = 'TCGA or Internal center, maybe another database can be used if you have', type = str)

    parser.add_argument("--save_dir", default = '/mnt/disk1/Kidney/Submission_dir/Figure2/final_data/', help = 'Directory to save the feature',type = str)
    parser.add_argument("--target_data", default = 'internal', help = 'Target_data name', type = str)
    parser.add_argument("--stainnorm", default = False, help = 'Stainnorm data or not', type = bool)
    parser.add_argument("--cancer_annotation", default = False, help = 'Use cancer annotation to draw', type = bool)
    parser.add_argument("--multi_scanner", default = True, help = 'Scanner comparison', type = bool)
    parser.add_argument("--with_normal", default = False, help = 'draw normal part in UMAP', type = bool)
    
    return parser.parse_args()

def main():
    Argument = Parser_main()
    draw_umap_and_calculate_NMI(Argument.metadata, Argument.target_center,  Argument.pfm_list, Argument.save_dir, Argument.target_data, Argument.stainnorm, Argument.cancer_annotation, Argument.multi_scanner, Argument.with_normal)
        
if __name__ == "__main__":
    main()
