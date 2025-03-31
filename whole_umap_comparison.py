import os
import torch
import umap
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
import openslide
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
import random
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
import matplotlib.patches as patches
from openslide import OpenSlide
import json
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score

def compare_silhouette_by_type_center(
    feature_list,  # list of torch.Tensor, each shape (d,) 혹은 (1,d)
    type_list,     # list or array-like, shape (N,)
    center_list,
    race_list,   # list or array-like, shape (N,)
    n_boot=5000,
    random_state=None
):
    """
    feature_list: 길이 N의 파이썬 리스트, 각 원소가 torch.Tensor (예: shape (d,))
    type_list: 길이 N의 라벨(Type) 목록
    center_list: 길이 N의 라벨(Center) 목록
    n_boot: 반복 샘플링 횟수 (int)
    random_state: 난수 시드 (int, optional)

    Returns:
        mean_sil_type, std_sil_type,
        mean_sil_center, std_sil_center
        (각각 Type/Center 기준 silhouette_score의 평균과 표준편차)
    """
    rng = np.random.default_rng(seed=random_state)  # numpy 1.17+ 스타일 난수 생성기

    # 1) feature_list (list of torch.Tensor) -> 하나의 2D Tensor로 스택
    #    만약 각 텐서가 shape (d,)라면 아래와 같이 stack 가능
    feature_tensor = torch.stack(feature_list, dim=0)  # shape: (N, d)
    
    # 필요하다면 GPU -> CPU 변환
    if feature_tensor.is_cuda:
        feature_tensor = feature_tensor.cpu()

    # 2) torch.Tensor -> numpy.ndarray 변환
    #    (autograd에 영향이 없도록 detach())
    feature_array = feature_tensor.detach().numpy()  # shape: (N, d)
    
    # 라벨들을 numpy array 형태로 맞춤
    type_arr = np.asarray(type_list)
    center_arr = np.asarray(center_list)
    race_arr = np.asarray(race_list)

    # 3) (type, center) 유니크 조합 찾기
    unique_types = np.unique(type_arr)
    unique_centers = np.unique(center_arr)
    unique_races = np.unique(race_arr)

    # 4) 각 (type, center) 조합별 인덱스 모으기
    group_dict = {}
    for t in unique_types:
        for c in unique_centers:
            for r in unique_races:
                idxs = np.where((type_arr == t) & (center_arr == c) & (race_arr == r))[0]
                if len(idxs) > 0:  # 실제로 샘플이 있는 조합만 저장
                    group_dict[(t, c, r)] = idxs

    # 5) 존재하는 (type, center) 조합 중 최소 크기 찾기
    min_count = min(len(idxs) for idxs in group_dict.values())

    # 6) 반복 샘플링 결과 저장용 리스트
    sil_scores_type = []
    sil_scores_center = []
    sil_scores_race = []

    for _ in range(n_boot):
        sampled_indices = []

        # 각 (type, center) 그룹에서 min_count만큼 무작위로 추출
        for (t, c, r), idxs in group_dict.items():
            chosen = rng.choice(idxs, size=min_count, replace=False)  # down-sampling
            sampled_indices.extend(chosen)

        sampled_indices = np.array(sampled_indices)

        # 서브샘플링된 데이터와 라벨
        X_sub = feature_array[sampled_indices]
        type_sub = type_arr[sampled_indices]
        center_sub = center_arr[sampled_indices]
        race_sub = race_arr[sampled_indices]

        # 7) Silhouette Score (기본 metric='euclidean' => L2 거리)
        sil_type = silhouette_score(X_sub, type_sub, metric='euclidean')
        sil_center = silhouette_score(X_sub, center_sub, metric='euclidean')
        sil_race = silhouette_score(X_sub, race_sub, metric = 'euclidean')

        sil_scores_type.append(sil_type)
        sil_scores_center.append(sil_center)
        sil_scores_race.append(sil_race)

    # 8) 반복 결과 요약(평균, 표준편차)
    mean_sil_type = np.mean(sil_scores_type)
    std_sil_type = np.std(sil_scores_type)
    mean_sil_center = np.mean(sil_scores_center)
    std_sil_center = np.std(sil_scores_center)
    mean_sil_race = np.mean(sil_scores_race)
    std_sil_race = np.std(sil_scores_race)

    return mean_sil_type, std_sil_type, mean_sil_center, std_sil_center, mean_sil_race, std_sil_race

def draw_umap(umap_data, umap_dir, type_list, sample_name_list, name, type_dict):
    umap_figure_dir = os.path.join(umap_dir, 'figure')
    os.makedirs(umap_figure_dir, exist_ok=True)
    umap_data_dir = os.path.join(umap_dir, 'data')
    os.makedirs(umap_data_dir, exist_ok=True)

    pdf_dir = os.path.join(umap_figure_dir, 'pdf')
    svg_dir = os.path.join(umap_figure_dir, 'svg')
    png_dir = os.path.join(umap_figure_dir, 'png')

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Convert type_list (numbers) to categorical names using the type_dict
    df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2']) 
    df['Sample_Name'] = sample_name_list    
    df['Type'] = type_list
    df.to_csv(os.path.join(umap_data_dir, f'umap_data_{name}.csv'), index=False)

    hue_order = [t for t in list(type_dict.keys()) if t in df['Type'].tolist()]
    
    # 10개의 tab10 컬러 가져오기
    tab10_colors = plt.get_cmap("tab20").colors[:20]

    # 레이블 목록 (예제)
    labels = ["KICH", "KIRC", "KIRP", "MD_Anderson_center", "BORAMAE", 'KHU', 'SNUBH', 'GSYUHS', "MSKCC", 'GSYUHS_leica', "NCI_Urologic", "asian", "white", "black or african american"]

    # 특정 레이블을 제외한 10개에 tab10 컬러 할당
    palette = {label: color for label, color in zip(labels[:15], tab10_colors)}

    # 특정 레이블에 black 할당
    palette["not reported"] = "black"

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Type', data=df, palette=palette, s=100, edgecolor=None, hue_order=hue_order, alpha = 0.7)
    
    # Set plot titles and labels
    plt.title(f'UMAP of Kidney Cancer')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    #  Save the plot as SVG (scalable vector format)
    svg_path = os.path.join(svg_dir, f'UMAP_of_{name}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)

    # Save the plot as PDF (alternative vector format)
    pdf_path = os.path.join(pdf_dir, f'UMAP_of_{name}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)

    # Save as a high-resolution PNG (raster format, for web or other purposes)
    png_path = os.path.join(png_dir, f'UMAP_of_{name}.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

def draw_umap_per_sample(umap_data, type_list , prototype_umap, prototype_type, umap_dir, name, TCGA_subtype):
    # Convert type_list (numbers) to categorical names using the type_dict
    df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2']) 
    df['Sample_Name'] = name    
    df['Type'] = type_list
    prototype_df = pd.DataFrame(prototype_umap, columns=['UMAP1', 'UMAP2'])
    prototype_df['Sample_Name'] = 'prototype'    
    prototype_df['Type'] = prototype_type
    final_df = pd.concat((df, prototype_df))

    final_type = list(set(type_list + prototype_type))

    hue_order = [t for t in final_type if t in final_df['Type'].tolist()]
    if ('center' not in name) and ('cluster' not in name):
        palette = {'KIRC_prototype': 'tab:orange', 'KIRC_cancer': 'tab:orange', 'KIRC': 'bisque', 'KIRC_additional': 'tab:orange',
                'KIRP_prototype': 'tab:green', 'KIRP_cancer': 'tab:green', 'KIRP': 'lightgreen', 'KIRP_additional': 'tab:green',
                'KICH_prototype': 'tab:blue', 'KICH_cancer': 'tab:blue', 'KICH': 'lightsteelblue', 'KICH_additional': 'tab:blue',
                'normal': 'grey', 'normal_prototype': 'black', 'Kidney_others': 'purple', 'Kidney_others_additional': 'purple', 'Translocation_RCC_prototype': 'tab:red', 'Translocation_RCC': 'lightcoral'}
    else:
        palette = 'tab20'
        
    # Plot using seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Type', data=final_df, palette=palette, s=10, edgecolor=None, hue_order=hue_order)
    
    # Set plot titles and labels
    plt.title(f'UMAP of {TCGA_subtype}_{name}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(umap_dir, f'{name}_UMAP.png'), bbox_inches='tight')
    plt.clf()
    plt.close()

def UMAP_drawing_and_bias_calculation(df, save_dir, feature_type, sampling_number = 0):
    random_seed = 123456
    whole_cancer_root = "/mnt/disk2/TEAgraph_preprocessing/"
    
    if 'stainnorm' in feature_type:
        feature_type_name = os.path.join(feature_type, "TCGA-B0-4827-01Z-00-DX1.c08eeafe-d2d4-41d1-a8f5-eacd1b0e601f")
    else:
        feature_type_name = feature_type
    prototype_dict = {}
    prototype_dict_whole = {}
    whole_feature_list = []
    whole_type_list = []
    whole_center_list = []
    whole_race_list = []
    df = df.drop_duplicates()
    sample_list = df['sample'].to_list()
    with tqdm(total = len(df)) as pbar:
        for i, target_df in df.iterrows():
            center = target_df['center']
            race = target_df['race']
            TCGA_subtype = target_df['TCGA_subtype']
            correct_subtype = target_df['correct_subtype']
            sample = target_df['sample']
            
            
            whole_cancer_embedding_path = os.path.join(whole_cancer_root, 'TCGA', TCGA_subtype, magnification, target_patch_size, feature_type_name, 'feature', 'pt_files', sample + '.pt')
            
            if os.path.exists(whole_cancer_embedding_path):
                whole_cancer_embedding = torch.load(whole_cancer_embedding_path)
            else:
                if center == 'BORAMAE':
                    whole_cancer_embedding_path = os.path.join('/mnt/disk2/TEAgraph_preprocessing/', center, TCGA_subtype, magnification, target_patch_size, feature_type_name, 'feature',  'pt_files', sample + '.pt')
                else:
                    whole_cancer_embedding_path = os.path.join('/mnt/disk2/TEAgraph_preprocessing/', center, 'RCC', magnification, target_patch_size, feature_type_name, 'feature',  'pt_files', sample + '.pt')
                whole_cancer_embedding = torch.load(whole_cancer_embedding_path)
            
            whole_cancer_embedding = torch.mean(whole_cancer_embedding, axis = 0)
            whole_feature_list.append(whole_cancer_embedding)
            whole_type_list.append(correct_subtype)
            whole_center_list.append(center)
            whole_race_list.append(race)

            if correct_subtype not in prototype_dict_whole.keys():
                prototype_dict_whole[correct_subtype] = [whole_cancer_embedding]
            else:
                prototype_dict_whole[correct_subtype].append(whole_cancer_embedding)
            pbar.update()
    
    # final_data = torch.stack(feature_list, axis = 0).numpy()
    whole_final_data = torch.stack(whole_feature_list, axis = 0).numpy()
    mean_t_whole, std_t_whole, mean_c_whole, std_c_whole, mean_r_whole, std_r_whole = compare_silhouette_by_type_center(
        whole_feature_list,
        whole_type_list,
        whole_center_list,
        whole_race_list,
        n_boot=5000,        # 예: 500번 반복
        random_state=1234)
        
    whole_reducer = umap.UMAP(n_components=2, random_state=random_seed)
    whole_umap_data = whole_reducer.fit_transform(whole_final_data)

    reducer = umap.UMAP(n_components=2, random_state=random_seed)
    # umap_data = reducer.fit_transform(final_data)
    umap_figure_dir = os.path.join(save_dir, 'UMAP')
    if os.path.exists(umap_figure_dir) is False:
        os.mkdir(umap_figure_dir)
    
    type_dict = {}
    for i, t in enumerate(list(set(whole_type_list))):
        type_dict[t] = i
    
    race_dict = {}
    for i, c in enumerate(list(set(whole_race_list))):
        race_dict[c] = i
    
    center_dict = {}
    for i, c in enumerate(list(set(whole_center_list))):
        center_dict[c] = i
    
    draw_umap(whole_umap_data, umap_figure_dir, whole_type_list, sample_list, f'subtype_{feature_type}_whole', type_dict)
    draw_umap(whole_umap_data, umap_figure_dir, whole_center_list, sample_list, f'center_{feature_type}_whole', center_dict)
    draw_umap(whole_umap_data, umap_figure_dir, whole_race_list, sample_list, f'race_{feature_type}_whole', race_dict)

    return prototype_dict, prototype_dict_whole, reducer, mean_t_whole, std_t_whole, mean_c_whole, std_c_whole, mean_r_whole, std_r_whole

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

Kidney_df = pd.read_csv("/mnt/disk1/Kidney/Final_analysis/Kidney_metadata_with_race.csv")
non_prototype_center = ['KHU', 'SNUBH', 'Dartmouth', 'GSYUHS', 'GSYUHS_leica']
remove_center = ''

feature_type_list = [ 'virchow', 'UNI', 'ProvGiga', 'virchow2', 'UNI2', 'CONCH', 'virchow_stainnorm', 'UNI_stainnorm', 'ProvGiga_stainnorm', 'virchow2_stainnorm', 'UNI2_stainnorm', 'CONCH_stainnorm']
patch_size_list = [ '224', '224', '256', '224', '224', '224', '224', '224', '256', '224', '224', '224']
magnification_list = [ '20x', '20x', '20x', '20x', '20x', '20x', '20x', '20x', '20x', '20x', '20x', '20x']

final_prediction_df = pd.DataFrame()


bias_name = 'stainnormß'
save_dir = "/mnt/disk1/Kidney/Final_Final_analysis/Analysis/Prototype_UMAP_stainnorm/"
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

target_type = ['KIRC', 'KICH', 'KIRP']
bias_dict = {}

for feature_type, target_patch_size, magnification in zip(feature_type_list, patch_size_list, magnification_list):
    print(feature_type)
    whole_cancer = "/mnt/disk1/Kidney/patch_level_analysis/dataset/20x/whole_cancer/"

    whole_cancer_root = os.path.join(whole_cancer, feature_type, target_patch_size)
    slide_root = "/mnt/disk2/TEAgraph_preprocessing/"
    
    target_type_sample = Kidney_df[Kidney_df['TCGA_subtype'].isin(target_type)]
    target_type_sample = target_type_sample[target_type_sample['center'] != remove_center]
    target_type_sample = target_type_sample[target_type_sample['center'] != 'Dartmouth']
    target_type_sample = target_type_sample[target_type_sample['correct_subtype'].isin(target_type)]
    target_type_sample = target_type_sample[target_type_sample['race'] != 'not reported']

    save_dir_feature = os.path.join(save_dir, feature_type)
    os.makedirs(save_dir_feature, exist_ok= True)

    prototype_dict, prototype_dict_whole, reducer, mean_t_whole, std_t_whole, mean_c_whole, std_c_whole, mean_r_whole, std_r_whole = UMAP_drawing_and_bias_calculation(target_type_sample, save_dir_feature, feature_type)
    bias_dict[feature_type + '_whole'] = [mean_t_whole, std_t_whole, mean_c_whole, std_c_whole, mean_r_whole, std_r_whole]

rows = []
for feat in bias_dict.keys():
    mean_t, std_t, mean_c, std_c, mean_r, std_r = bias_dict[feat]
    rows.append({
        "Feature": feat,
        "Group": "Type",
        "Mean": mean_t,
        "Std": std_t
    })
    rows.append({
        "Feature": feat,
        "Group": "Center",
        "Mean": mean_c,
        "Std": std_c
    })
    rows.append({
        "Feature": feat,
        "Group": "Race",
        "Mean": mean_r,
        "Std": std_r
    })

bias_root_dir = "/mnt/disk1/Kidney/Final_Final_analysis/Analysis/bias_figure/"
bias_save_dir = os.path.join(bias_root_dir, bias_name)
os.makedirs(bias_save_dir, exist_ok = True)


df = pd.DataFrame(rows)
df.to_csv(os.path.join(bias_save_dir, "bias_dataframe_not_abs.csv"), index = False)

df = pd.read_csv(os.path.join(bias_save_dir, "bias_dataframe_not_abs.csv"))
center_bias_df = df[df['Group'] == 'Center']
type_bias_df = df[df['Group'] == 'Type']
race_bias_df = df[df['Group'] == 'Race']

# 범위 설정
y_min = df["Mean"].min() - 0.05
y_max = df["Mean"].max() + 0.05

# Bar plot 그리기
fig, ax = plt.subplots(figsize=(8, 6))

# 그룹별 bar width 설정
bar_width = 0.4
features = df["Feature"].unique()
groups = df["Group"].unique()
x = np.arange(len(features))

# 각 그룹별 막대 그래프 그리기
for i, group in enumerate(groups):
    group_data = df[df["Group"] == group]
    ax.bar(
        x + (i - len(groups)/2) * bar_width,
        group_data["Mean"],
        yerr=group_data["Std"],
        width=bar_width,
        capsize=5,
        label=group
    )

# x축 설정
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.set_xlabel("Feature")
ax.set_ylabel("Mean Value")
ax.set_ylim(y_min, y_max)
ax.legend(title="Group")
ax.set_title("Bar Plot with Mean and Std")

plt.savefig(os.path.join(os.path.join(bias_save_dir, "bias_barplot_whole.png")))
plt.close()

# Seaborn을 사용한 bar plot 그리기
plt.figure(figsize=(8, 6))
ax = sns.barplot(
    data=df,
    x="Group",
    y="Mean",
    hue="Feature",
    ci=None,  # ci=None으로 설정하여 seaborn의 기본 신뢰구간 제거
    capsize=0.1
)

# 표준편차를 error bar로 추가
for i, group in enumerate(groups):
    group_data = df[df["Group"] == group]
    x_positions = [p.get_x() + p.get_width() / 2 for p in ax.patches[i::len(groups)]]

# y축 범위 설정
ax.set_ylim(y_min, y_max)

# 그래프 설정
ax.set_xlabel("Feature")
ax.set_ylabel("Mean Value")
ax.set_title("Bar Plot with Mean and Std (Seaborn)")
plt.legend(title="Group")

plt.savefig(os.path.join(os.path.join(bias_save_dir, "bias_barplot_whole_seaborn.png")))
plt.close()

plt.figure(figsize=(12, 6))
# Barplot with 'Feature' on x-axis, 'Mean' as the height, and 'Group' as hue
sns.barplot(
    data=center_bias_df, 
    x='Feature', 
    y='Mean', 
    dodge=True,  # 그룹별 막대 분리
    palette='Set2'
)

plt.xlabel("Feature")
plt.ylabel("Center Bias mean")
plt.title("Center Bias")
plt.legend(title="Legend", loc='upper right')
plt.xticks(rotation=45)
plt.ylim((0,0.25))
plt.tight_layout()

plt.savefig(os.path.join(os.path.join(bias_save_dir, "center_bias_with_BORAMAE_not_abs.png")))
plt.close()

plt.figure(figsize=(12, 6))

# Barplot with 'Feature' on x-axis, 'Mean' as the height, and 'Group' as hue
sns.barplot(
    data=type_bias_df, 
    x='Feature', 
    y='Mean',  
    dodge=True,  # 그룹별 막대 분리
    palette='Set2'
)

plt.xlabel("Feature")
plt.ylabel("Type Bias mean")
plt.title("Type Bias")
plt.legend(title="Legend", loc='upper right')
plt.xticks(rotation=45)
plt.ylim((0,0.25))
plt.tight_layout()

plt.savefig(os.path.join(os.path.join(bias_save_dir, "type_bias_with_BORAMAE_not_abs.png")))
plt.close()

plt.figure(figsize=(12, 6))

# Barplot with 'Feature' on x-axis, 'Mean' as the height, and 'Group' as hue
sns.barplot(
    data=race_bias_df, 
    x='Feature', 
    y='Mean',  
    dodge=True,  # 그룹별 막대 분리
    palette='Set2'
)

plt.xlabel("Feature")
plt.ylabel("Race Bias mean")
plt.title("Race Bias")
plt.legend(title="Legend", loc='upper right')
plt.xticks(rotation=45)
plt.ylim((0,0.25))
plt.tight_layout()

plt.savefig(os.path.join(os.path.join(bias_save_dir, "race_bias_with_BORAMAE_not_abs.png")))
plt.close()

