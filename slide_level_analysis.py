import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
import argparse

def compute_SDD_from_long(
    dist_long_df: pd.DataFrame,
    sample_col: str = "sample",
    label_col: str = "center",    
    proto_col: str = "center_proto", 
    dist_col: str = "distance",
    trim_q: float = None,   # e.g. 0.95 for tail trimming
):
    """
    Compute SDD (mean Wasserstein distance) from long-format distance CSV.
    """
    out = []

    samples = dist_long_df[sample_col].unique()
    all_protos = dist_long_df[proto_col].unique()

    for s in tqdm(samples):
        sdf = dist_long_df[dist_long_df[sample_col] == s]
        true_label = sdf[label_col].iloc[0]

        # true prototype name
        true_proto = [p for p in all_protos if true_label in p]
        if len(true_proto) != 1:
            out.append({"Sample": s, f"SDD_{label_col}": np.nan})
            continue
        true_proto = true_proto[0]

        d_true = sdf[sdf[proto_col] == true_proto][dist_col].values
        if trim_q is not None:
            hi = np.quantile(d_true, trim_q)
            d_true = d_true[d_true <= hi]

        wds = []
        w_dict = {"Sample": s, "label": true_proto}
        for p in all_protos:
            if p == true_proto:
                w_dict[f"SDD_{p}"] = 0
                continue
            d_other = sdf[sdf[proto_col] == p][dist_col].values
            if trim_q is not None:
                hi = np.quantile(d_other, trim_q)
                d_other = d_other[d_other <= hi]
            if len(d_true) == 0 or len(d_other) == 0:
                continue
            wd = wasserstein_distance(d_true, d_other)
            wds.append(wd)
            w_dict[f"SDD_{p}"] = wd

        sdd = float(np.mean(wds)) if len(wds) > 0 else np.nan
        w_dict[f"SDD_{label_col}"] = sdd
        out.append(w_dict)

    return pd.DataFrame(out)

def slide_level_analysis(root_dir, meta_data, pfm_list, save_dir, target_column, stainnorm = False, cancer_only = False):
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
    sdd_dir = os.path.join(save_dir, 'sdd')
    os.makedirs(sdd_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    samplewise_sdd_dir = os.path.join(save_dir, 'samplewise_sdd')
    os.makedirs(samplewise_sdd_dir, exist_ok = True)

    for pfm in pfm_list:
        if os.path.exists(os.path.join(sdd_dir, '%s_%s_sdd.csv' % (pfm, target_column))):
            sdd_df = pd.read_csv(os.path.join(sdd_dir, '%s_%s_sdd.csv' % (pfm, target_column)))
        else:
            target_df = pd.read_csv(os.path.join(dist_dir, '%s_dist_to_%s_prototypes_long.csv' % (pfm, target_column)))
            sdd_df = compute_SDD_from_long(target_df, label_col= target_column, proto_col=target_column + '_proto')
            sdd_df.to_csv(os.path.join(sdd_dir, '%s_%s_sdd.csv' % (pfm, target_column)), index=False)
        
    total_df = []
    for pfm in pfm_list:
        df = pd.read_csv(os.path.join(sdd_dir, '%s_%s_sdd.csv' % (pfm, target_column)))
        df['PFM'] = pfm
        total_df.append(df)
        
    total_df = pd.concat(total_df)
    
    plt.figure()
    sns.barplot(data=total_df, x='PFM', y='SDD_%s' % (target_column), order=pfm_list)
    plt.ylim(0, 0.2)
    plt.savefig(os.path.join(plot_dir, '%s_sdd_%s_barplot.png' % (data_type, target_column)))
    plt.savefig(os.path.join(plot_dir, '%s_sdd_%s_barplot.svg' % (data_type, target_column)), dpi=1000)
    
    # hue_order = [target_column + '__' + c for c in target_center]

    plt.figure()
    sns.barplot(data=total_df, x='PFM', y='SDD_%s' % target_column, hue='label', order=pfm_list)
    plt.ylim(0, 0.4)

    plt.savefig(os.path.join(plot_dir, '%s_psi_%s_barplot_per_type.png' % (data_type, target_column)))
    plt.savefig(os.path.join(plot_dir, '%s_psi_%s_barplot_per_type.svg' % (data_type, target_column)), dpi=1000)
    
    
    feature_results = []
    sample_meta_df = meta_df[['Patient', 'subtype', 'center']].rename(columns = {'Patient': 'Sample'})
    for pfm in pfm_list:
        base_df = pd.read_csv(
            os.path.join(sdd_dir, f'{pfm}_{target_column}_sdd.csv'),
            usecols=['Sample', 'label', 'SDD_' + target_column]
        )
        xmin, xmax = (min(base_df['SDD_' + target_column]), max(base_df['SDD_' + target_column]))
        df = pd.read_csv(
            os.path.join(sdd_dir, f'{pfm}_{target_column}_sdd.csv'),
            usecols=['Sample', 'label', 'SDD_' + target_column]
        )

        df['SDD_norm'] = (
            df['SDD_' + target_column]
            .transform(lambda x: (x - xmin) / (xmax - xmin))
        )
        df['PFM'] = pfm
        df = pd.merge(df, sample_meta_df, how = 'left', on = 'Sample')
        feature_results.append(df)
        
    total_df = pd.concat(feature_results)

    total_df = total_df.sort_values(['center', 'subtype'])
    sample_order = total_df[total_df['PFM']==pfm_list[0]]['Sample'].tolist()

    total_df.to_csv(os.path.join(samplewise_sdd_dir, '%s_samplewise_SDD_%s_total.csv' % (data_type, target_column)), index=False)
    pivot = total_df[['Sample', 'PFM', 'SDD_norm']].pivot(index='PFM', columns='Sample', values='SDD_norm')
    pivot = pivot.loc[pfm_list, sample_order]
    pivot.to_csv(os.path.join(samplewise_sdd_dir, '%s_samplewise_SDD_%s.csv' % (data_type, target_column)))
    plt.figure()
    sns.heatmap(pivot, cmap='Reds')
    plt.savefig(os.path.join(plot_dir, '%s_samplewise_SDD_mean_%s_heatmap.png' % (data_type, target_column)))
    plt.savefig(os.path.join(plot_dir, '%s_samplewise_SDD_mean_%s_heatmap.svg' % (data_type, target_column)), dpi=1000)

    
    
def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--root_dir", default = "/mnt/disk1/Kidney/Github_test/prototype_distance_dataset/", help = 'root_dir of analysis', type = str, required = False)
    parser.add_argument("--metadata", default = "/mnt/disk1/Kidney/Github_test/metadata.csv", help = 'Path of the metadata', type = str, required = False)
    parser.add_argument("--save_dir", default = "/mnt/disk1/Kidney/Github_test/slide_level_analysis/", help = 'Directory to save the feature',type = str, required = False)
    parser.add_argument("--pfm_list", nargs = "+", default = ['virchow', 'virchow2', 'UNI', 'UNI2'], help = 'PFM list for comparison', type = str)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--stainnorm", default = False, help = 'Stainnorm data or not', type = bool)
    parser.add_argument("--cancer_only", default = False, help = 'Use cancer only data', type = bool)
    
    return parser.parse_args()

def main():
    Argument = Parser_main()
    slide_level_analysis(Argument.root_dir, Argument.metadata, Argument.pfm_list, Argument.save_dir, Argument.target_column, Argument.stainnorm, Argument.cancer_only)
        
if __name__ == "__main__":
    main()
