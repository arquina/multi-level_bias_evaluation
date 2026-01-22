import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import (
    shapiro,
    ttest_rel,
    ttest_ind,
    wilcoxon,
    mannwhitneyu,
    levene
)

rootdir = '/mnt/disk1/Kidney/Submission_dir/distance_analysis_original_prototype_TCGA'

mitigations = ['original', 'stainnorm', 'canceronly', 'stainnorm_canceronly']
feature_origin_list = ['virchow', 'virchow2', 'UNI', 'UNI2', 'GigaPath', 'CONCH'] 

center_dict = {'KIRP': {'BQ': 'MSKCC', 'DW': 'NCI', 'P4': 'MDACC'},
            'KIRC': {'BP': 'MSKCC', 'DV': 'NCI', 'CJ': 'MDACC'},
            'KICH': {'KL': 'MSKCC', 'KM': 'NCI', 'KO': 'MDACC'}}

patch_size_dict = {'virchow': 224, 'virchow2': 224, 'UNI': 256, 'UNI2': 256, 'GigaPath': 256, 'CONCH': 512, }

def select_significance(list_1, list_2, paired=False, alpha=0.05):
    list_1 = np.asarray(list_1)
    list_2 = np.asarray(list_2)

    n1, n2 = len(list_1), len(list_2)

    # 최소 샘플 수
    if n1 < 5 or n2 < 5:
        return None, None

    # ------------------
    # Paired case
    # ------------------
    if paired:
        if n1 != n2:
            raise ValueError("Paired test requires equal-length samples.")

        diff = list_1 - list_2
        n = len(diff)

        # large sample: skip normality test
        if n >= 30:
            stat, p = ttest_rel(list_1, list_2)
            return "paired t-test", p

        # small sample: test normality of difference
        if shapiro(diff)[1] > alpha:
            stat, p = ttest_rel(list_1, list_2)

            return "paired t-test", p
        else:
            stat, p = wilcoxon(list_1, list_2)
            return "wilcoxon signed-rank", p

    # ------------------
    # Unpaired case
    # ------------------
    else:
        # large sample: Welch is safest
        if n1 >= 30 and n2 >= 30:
            stat, p = ttest_ind(list_1, list_2, equal_var=False)
            return "welch t-test", p

        # small sample: check normality
        normal_1 = shapiro(list_1)[1] > alpha
        normal_2 = shapiro(list_2)[1] > alpha

        if normal_1 and normal_2:
            # check variance equality
            if levene(list_1, list_2)[1] > alpha:
                stat, p = ttest_ind(list_1, list_2, equal_var=True)
                return "t-test_ind", p
            else:
                stat, p = ttest_ind(list_1, list_2, equal_var=False)
                return "welch t-test", p
        else:
            stat, p = mannwhitneyu(list_1, list_2, alternative="two-sided")
            return "mann-whitney", p
        
        
if True:
    def compute_psi_from_wide(
        dist_wide_df: pd.DataFrame,
        label_col: str = "Center",     # or "Subtype"
        label_list: list = [],
    ):
        """
        Compute PSI (mean Wasserstein distance) from long-format distance CSV.
        """
        out = []

        for i in range(len(dist_wide_df)):
            sdf = dist_wide_df.iloc[i]
            true_proto = '%s__%s' % (str.upper(label_col), sdf[label_col])
            all_protos = ['%s__%s' % (str.upper(label_col), l) for l in label_list]
            sample = sdf['Sample']
            psi_list = []
            w_dict = {"Sample": sample, "label": true_proto}
            d_true = sdf[true_proto]
            for p in all_protos:
                if p == true_proto:
                    w_dict[f"PSI_{p}"] = 0
                    continue
                d_other = sdf[p]

                psi = np.abs(d_true-d_other)
                psi_list.append(psi)
                w_dict[f"PSI_{p}"] = psi

            psi = float(np.mean(psi_list)) if len(psi_list) > 0 else np.nan
            w_dict[f"PSI_{label_col}"] = psi
            out.append(w_dict)
        return pd.DataFrame(out)
    
    target = 'subtype'
    for m in mitigations[:1]:
        for f in feature_origin_list:
            df = pd.read_csv(os.path.join(rootdir, m, 'distance', '%s_dist_to_%s_prototypes_wide.csv' % (f, target)))
            label_col = str.upper(target[0])+target[1:]
            psi_df = compute_psi_from_wide(dist_wide_df=df, label_col=label_col, label_list=df[label_col].unique())
            print(psi_df)
            psi_df.to_csv(os.path.join(rootdir, m, 'distance', '%s_%s_psi.csv' % (f, target)), index=False)


if True:
    target = 'center'
    label_col = str.upper(target[0])+target[1:]
    psi_col = 'PSI_%s' % label_col
    savedir = os.path.join(rootdir, 'plot', 'patch')
    if os.path.exists(savedir) is False: os.mkdir(savedir)
    
    for m in mitigations[:1]:
        total_df = []
        for f in feature_origin_list:
            df = pd.read_csv(os.path.join(rootdir, m, 'distance', '%s_%s_psi.csv' % (f, target)))
            df['feature'] = f
            total_df.append(df)
        total_df = pd.concat(total_df)
        
        plt.figure()
        sns.boxplot(data=total_df, x='feature', y='PSI_%s' % label_col, showfliers=False)
        plt.ylim(-0.02, 0.3)
        plt.savefig(os.path.join(savedir, '%s_%s_psi_boxplot.png' % (target, m)))
        plt.savefig(os.path.join(savedir, '%s_%s_psi_boxplot.svg' % (target, m)), dpi=1000)
      
if True:
    target = 'center'
    label_col = str.upper(target[0])+target[1:]
    psi_col = 'PSI_%s' % label_col
    savedir = os.path.join(rootdir, 'plot', 'patch')
    if os.path.exists(savedir) is False: os.mkdir(savedir)
    palette=['lightpink', 'cornflowerblue']
    for m in mitigations[:1]:
        total_df = []
        for f in feature_origin_list:
            df = pd.read_csv(os.path.join(rootdir, m, 'distance', '%s_%s_psi.csv' % (f, target)))
            high_threshold = df[psi_col].quantile(0.75)
            df['is_high'] = df[psi_col] > high_threshold
            df['feature'] = f
            total_df.append(df)
        total_df = pd.concat(total_df)[['is_cancer', 'feature', 'Sample', 'is_high']]

        ratio_df = (
            total_df
            .groupby(["feature", "Sample", "is_cancer"])
            .agg(
                high_count=("is_high", "sum"),   # high threshold 이상 개수
                total_count=("is_high", "count") # 전체 개수
            )
            .reset_index()
        )

        ratio_df["ratio"] = ratio_df["high_count"] / ratio_df["total_count"]
        ratio_df['subtype'] = [s for sample in ratio_df['Sample'] for s in center_dict.keys() if sample.split('-')[1] in center_dict[s]]
        ratio_df['center'] = [center_dict[s][sample.split('-')[1]] for s, sample in zip(ratio_df['subtype'], ratio_df['Sample'])]
        ratio_df['label'] = ['_'.join([c, s]) for c, s in zip(ratio_df['center'], ratio_df['subtype'])]
        
        print(ratio_df)
        ratio_df.to_csv(os.path.join(savedir, 'high_PSI_cancer_normal_ratio.csv'), index=False)

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=ratio_df, x='feature', y='ratio', hue='is_cancer', order=feature_origin_list, palette=palette, hue_order=[True, False])
        sns.stripplot(data=ratio_df, x='feature', y='ratio', hue='is_cancer', order=feature_origin_list, dodge=True, palette=palette, alpha=0.5, hue_order=[True, False])
        plt.savefig(os.path.join(savedir, '%s_%s_high_PSI_region_proportion_per_region_per_sample_boxplot.png' % (m, target)))
        plt.savefig(os.path.join(savedir, '%s_%s_high_PSI_region_proportion_per_region_per_sample_boxplot.svg' % (m, target)), dpi=1000)

if True:
    mitigation_basedir = os.path.join(rootdir, mitigations[0])
    savedir = os.path.join(rootdir, 'plot', 'patch')
    target = 'center'
    label_col = str.upper(target[0])+target[1:]
    psi_col = 'PSI_%s' % label_col
    maxy = [0.44, 0.11, 0.22, 0.22, 0.165, 0.11]
    maxy = [0.22, 0.088, 0.154, 0.154, 0.154, 0.088]
    for m in mitigations[1:2]:
        mitigation_dir = os.path.join(rootdir, m)

        fig, ax = plt.subplots(1, len(feature_origin_list), figsize=(30, 5))
        for i, f in enumerate(feature_origin_list):
            b_df = pd.read_csv(
                os.path.join(rootdir, mitigations[0], 'distance', f'{f}_{target}_psi.csv'),
                usecols=['Sample', psi_col]
            )
            m_df = pd.read_csv(
                os.path.join(rootdir, m, 'distance', f'{f}_{target}_psi.csv'),
                usecols=['Sample', psi_col]
            )

            b_df['mitigation'] = 'original'
            m_df['mitigation'] = m
            total_df = pd.concat([b_df, m_df])
            total_df = total_df.groupby(['mitigation', 'Sample'])[psi_col].mean().reset_index()
            print(total_df)
            test, pvalue = select_significance(b_df[psi_col], m_df[psi_col], paired=True)
            print(test, pvalue)
            # boxplot
            sns.boxplot(
                data=total_df,
                x="mitigation",
                y=psi_col,
                color="white",
                showcaps=True,
                boxprops={"edgecolor": "black"},
                whiskerprops={"color": "black"},
                medianprops={"color": "black"},
                showfliers=False,
                ax=ax[i]
            )

            # paired points + lines
            sns.stripplot(
                data=total_df,
                x="mitigation",
                y=psi_col,
                # hue="Sample",
                dodge=False,
                jitter=False,
                size=4,
                alpha=0.6,
                legend=False,
                ax=ax[i]
                
            )

            # connect paired points
            for sid, sub_df in total_df.groupby('Sample'):
                ax[i].plot(
                    [0, 1],
                    sub_df[psi_col],
                    color="gray",
                    alpha=0.4,
                    linewidth=1
                )
            ax[i].set_ylim(0, maxy[i])
            ax[i].set_title('%s_%s (%s: %.2e)' % (target, f, test, pvalue))
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, '%s_psi_%s_paired_boxplot.png' % (target, m)))             
        plt.savefig(os.path.join(savedir, '%s_psi_%s_paired_boxplot.svg' % (target, m)), dpi=1000)             
