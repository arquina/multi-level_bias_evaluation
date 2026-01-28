import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
        
def compute_pdd_from_wide(
    dist_wide_df: pd.DataFrame,
    label_col: str = "center",
    label_list: list = [],
):
    """
    Compute PDD from wide-format distance CSV.
    """
    out = []

    for i in range(len(dist_wide_df)):
        sdf = dist_wide_df.iloc[i]
        true_proto = '%s__%s' % (label_col, sdf[label_col])
        all_protos = ['%s__%s' % (label_col, l) for l in label_list]
        sample = sdf['sample']
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
            pdd_df = compute_pdd_from_wide(dist_wide_df=df, label_col=target_column, label_list=df[target_column].unique())
            pdd_df.to_csv(os.path.join(pdd_dir,'%s_%s_pdd.csv' % (pfm, target_column)), index=False)

    pdd_col = 'PDD_%s' % target_column
    
    total_df = []
    for pfm in pfm_list:
        df = pd.read_csv(os.path.join(pdd_dir, '%s_%s_pdd.csv' % (pfm, target_column)))
        df['PFM'] = pfm
        total_df.append(df)
    total_df = pd.concat(total_df)
    
    plt.figure()
    sns.boxplot(data=total_df, x='PFM', y='PDD_%s' % target_column, showfliers=False)
    plt.ylim(-0.02, 0.3)
    plt.savefig(os.path.join(plot_dir, '%s_%s_pdd_boxplot.png' % (target_column, data_type)))
    plt.savefig(os.path.join(plot_dir, '%s_%s_pdd_boxplot.svg' % (target_column, data_type)), dpi=1000)

    if 'is_cancer' in total_df.columns:
        palette=['lightpink', 'cornflowerblue']

        total_df = []
        sample_meta_df = meta_df[['Patient', 'subtype', 'center']].rename(columns = {'Patient': 'sample'})
        for pfm in pfm_list:
            df = pd.read_csv(os.path.join(pdd_dir, '%s_%s_pdd.csv' % (pfm, target_column)))
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

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=ratio_df, x='PFM', y='ratio', hue='is_cancer', order=pfm_list, palette=palette, hue_order=[True, False])
        sns.stripplot(data=ratio_df, x='PFM', y='ratio', hue='is_cancer', order=pfm_list, dodge=True, palette=palette, alpha=0.5, hue_order=[True, False])
        plt.savefig(os.path.join(plot_dir, '%s_%s_high_PDD_region_proportion_per_region_per_sample_boxplot.png' % (data_type, target_column)))
        plt.savefig(os.path.join(plot_dir, '%s_%s_high_PDD_region_proportion_per_region_per_sample_boxplot.svg' % (data_type, target_column)), dpi=1000)    

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--root_dir", help = 'root_dir of analysis', type = str, required = True)
    parser.add_argument("--metadata", help = 'Path of the metadata', type = str, required = True)
    parser.add_argument("--save_dir", help = 'Directory to save the feature',type = str, required = True)
    parser.add_argument("--pfm_list", nargs = "+", default = [], help = 'PFM list for comparison', type = str)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--stainnorm", default = False, help = 'Stainnorm data or not', type = bool)
    parser.add_argument("--cancer_only", default = False, help = 'Use cancer only data', type = bool)
    
    return parser.parse_args()

def main():
    Argument = Parser_main()
    patch_level_analysis(Argument.root_dir, Argument.metadata, Argument.pfm_list, Argument.save_dir, Argument.target_column, Argument.stainnorm, Argument.cancer_only)
        
if __name__ == "__main__":
    main()