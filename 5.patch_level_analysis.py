import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


def paired_significance(list_1, list_2, alpha=0.05):
    """Paired cancer-vs-normal significance for one PFM (same adaptive
    t-test/Wilcoxon selection used in 6.compare_mitigation.py, for
    methodological consistency across the pipeline)."""
    list_1 = np.asarray(list_1)
    list_2 = np.asarray(list_2)
    n = len(list_1)
    if n < 5 or n != len(list_2):
        return None, None

    diff = list_1 - list_2
    if n >= 30:
        _, p = ttest_rel(list_1, list_2)
        return "paired t-test", p
    if shapiro(diff)[1] > alpha:
        _, p = ttest_rel(list_1, list_2)
        return "paired t-test", p
    _, p = wilcoxon(list_1, list_2)
    return "wilcoxon signed-rank", p


def compute_pdd_from_wide(
    dist_wide_df: pd.DataFrame,
    sample_col: str = "Sample",
    label_col: str = "Center",
    label_list: list = [],
):
    """
    Compute PDD from wide-format distance CSV.
    """
    out = []

    for i in range(len(dist_wide_df)):
        sdf = dist_wide_df.iloc[i]
        true_proto = '%s__%s' % (str.upper(label_col), sdf[label_col])
        all_protos = ['%s__%s' % (str.upper(label_col), l) for l in label_list]
        sample = sdf[sample_col]
        pdd_list = []
        w_dict = {"sample": sample, "label": true_proto}
        d_true = sdf[true_proto]
        for p in all_protos:
            if p == true_proto:
                w_dict[f"PDD_{p}"] = 0
                continue
            d_other = sdf[p]

            pdd = np.abs(d_true-d_other)
            pdd_list.append(pdd)
            w_dict[f"PDD_{p}"] = pdd

        pdd = float(np.mean(pdd_list)) if len(pdd_list) > 0 else np.nan
        w_dict[f"PDD_{label_col}"] = pdd
        out.append(w_dict)
    return pd.DataFrame(out)

def patch_level_analysis(root_dir, meta_data, pfm_list, save_dir, target_column, stainnorm = False, cancer_only = False):
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
    
    meta_df = pd.read_csv(meta_data)
    meta_df= meta_df[meta_df['center']!='GSH_histech']

    mitigation_dir = os.path.join(root_dir, data_type)
    dist_dir = os.path.join(mitigation_dir, 'distance')

    pdd_dir = os.path.join(save_dir, 'pdd')
    os.makedirs(pdd_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)

    for pfm in pfm_list:
        if os.path.exists(os.path.join(pdd_dir,'%s_%s_pdd.csv' % (pfm, target_column))):
            pdd_df = pd.read_csv(os.path.join(pdd_dir,'%s_%s_pdd.csv' % (pfm, target_column)))
        else:
            df = pd.read_csv(os.path.join(dist_dir, '%s_dist_to_%s_prototypes_wide.csv' % (pfm, target_column)))
            pdd_df = compute_pdd_from_wide(dist_wide_df=df, label_col=target_column.capitalize(), label_list=df[target_column.capitalize()].unique())
            pdd_df.to_csv(os.path.join(pdd_dir,'%s_%s_pdd.csv' % (pfm, target_column)), index=False)

    pdd_col = 'PDD_%s' % target_column.capitalize()
    
    total_df = []
    for pfm in pfm_list:
        df = pd.read_csv(os.path.join(pdd_dir, '%s_%s_pdd.csv' % (pfm, target_column)))
        df['PFM'] = pfm
        total_df.append(df)
    total_df = pd.concat(total_df)
    
    plt.figure()
    sns.boxplot(data=total_df, x='PFM', y=pdd_col, showfliers=False)
    # plt.ylim(-0.02, 0.3)
    plt.savefig(os.path.join(plot_dir, '%s_%s_pdd_boxplot.png' % (target_column, data_type)))
    plt.savefig(os.path.join(plot_dir, '%s_%s_pdd_boxplot.svg' % (target_column, data_type)), dpi=1000)

    # total_df (built above from the plain *_pdd.csv files) never actually carries an
    # 'is_cancer' column -- compute_pdd_from_wide only emits sample/label/PDD_* columns.
    # What determines whether this analysis is possible is whether the precomputed
    # *_pdd_with_is_cancer.csv files exist on disk (from a separate upstream step).
    has_is_cancer_files = all(
        os.path.exists(os.path.join(pdd_dir, '%s_%s_pdd_with_is_cancer.csv' % (pfm, target_column)))
        for pfm in pfm_list
    )
    if not has_is_cancer_files:
        print(f"#Skipping cancer-vs-normal analysis: missing *_{target_column}_pdd_with_is_cancer.csv "
              f"for one or more of {pfm_list} in {pdd_dir}")
    if has_is_cancer_files:
        palette=['lightpink', 'cornflowerblue']

        total_df = []
        sample_meta_df = meta_df[['sample', 'subtype', 'center']]
        for pfm in pfm_list:
            df = pd.read_csv(os.path.join(pdd_dir, '%s_%s_pdd_with_is_cancer.csv' % (pfm, target_column)))
            high_threshold = df[pdd_col].quantile(0.75)
            df['is_high'] = df[pdd_col] > high_threshold
            df['PFM'] = pfm
            total_df.append(df)
        total_df = pd.concat(total_df)[['is_cancer', 'PFM', 'sample', 'is_high']]

        ratio_df = (
            total_df
            .groupby(["PFM", "sample", "is_cancer"])
            .agg(
                high_count=("is_high", "sum"),   # high threshold 이상 개수
                total_count=("is_high", "count") # 전체 개수
            )
            .reset_index()
        )

        ratio_df["ratio"] = ratio_df["high_count"] / ratio_df["total_count"]
        ratio_df = pd.merge(ratio_df, sample_meta_df, how = 'left', on = 'sample')
        ratio_df['label'] = ['_'.join([c, s]) for c, s in zip(ratio_df['center'], ratio_df['subtype'])]
        
        ratio_df.to_csv(os.path.join(plot_dir, 'high_PDD_cancer_normal_ratio.csv'), index=False)

        # cancer vs normal paired significance, per PFM (paired on 'sample'),
        # BH-FDR corrected across the PFMs shown together in this plot.
        stat_rows = []
        for pfm in pfm_list:
            wide = (
                ratio_df[ratio_df['PFM'] == pfm]
                .pivot(index='sample', columns='is_cancer', values='ratio')
                .dropna(subset=[True, False])
            )
            test, pvalue = paired_significance(wide[True], wide[False])
            stat_rows.append({'PFM': pfm, 'n_paired_samples': len(wide), 'test': test, 'pvalue': pvalue})
        stat_df = pd.DataFrame(stat_rows)
        valid = stat_df['pvalue'].notna()
        stat_df['pvalue_adj'] = np.nan
        if valid.any():
            stat_df.loc[valid, 'pvalue_adj'] = multipletests(stat_df.loc[valid, 'pvalue'], method='fdr_bh')[1]
        stat_df.to_csv(os.path.join(plot_dir, 'high_PDD_cancer_normal_ratio_pvalues.csv'), index=False)
        print(stat_df)

        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        sns.boxplot(data=ratio_df, x='PFM', y='ratio', hue='is_cancer', order=pfm_list, palette=palette, hue_order=[True, False], ax=ax)
        sns.stripplot(data=ratio_df, x='PFM', y='ratio', hue='is_cancer', order=pfm_list, dodge=True, palette=palette, alpha=0.5, hue_order=[True, False], ax=ax)

        y_top = ratio_df['ratio'].max()
        for i, pfm in enumerate(pfm_list):
            row = stat_df[stat_df['PFM'] == pfm].iloc[0]
            label = 'p_adj=%.2e' % row['pvalue_adj'] if pd.notna(row['pvalue_adj']) else 'n/a'
            ax.text(i, y_top * 1.03, label, ha='center', va='bottom', fontsize=8)
        ax.set_ylim(top=y_top * 1.15)

        plt.savefig(os.path.join(plot_dir, '%s_%s_high_PDD_region_proportion_per_region_per_sample_boxplot.png' % (data_type, target_column)))
        plt.savefig(os.path.join(plot_dir, '%s_%s_high_PDD_region_proportion_per_region_per_sample_boxplot.svg' % (data_type, target_column)), dpi=1000)

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--root_dir", help = 'root_dir of analysis', type = str, required = True)
    parser.add_argument("--metadata", help = 'Path of the metadata', type = str, required = True)
    parser.add_argument("--save_dir", help = 'Directory to save the feature',type = str, required = True)
    parser.add_argument("--pfm_list", nargs = "+", default = [], help = 'PFM list for comparison', type = str)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--stainnorm", action = 'store_true', help = 'Use stainnorm data instead of original')
    parser.add_argument("--cancer_only", action = 'store_true', help = 'Use cancer-only data')
    
    return parser.parse_args()

def main():
    Argument = Parser_main()
    patch_level_analysis(Argument.root_dir, Argument.metadata, Argument.pfm_list, Argument.save_dir, Argument.target_column, Argument.stainnorm, Argument.cancer_only)
        
if __name__ == "__main__":
    main()