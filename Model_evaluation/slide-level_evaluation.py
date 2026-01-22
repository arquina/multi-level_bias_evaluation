import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance


rootdir = '/mnt/disk1/Kidney/Submission_dir/distance_analysis_original_prototype_TCGA'

mitigations = ['original', 'stainnorm', 'canceronly', 'stainnorm_canceronly']

center_dict = {'KIRP': {'BQ': 'MSKCC', 'DW': 'NCI', 'P4': 'MDACC'},
            'KIRC': {'BP': 'MSKCC', 'DV': 'NCI', 'CJ': 'MDACC'},
            'KICH': {'KL': 'MSKCC', 'KM': 'NCI', 'KO': 'MDACC'}}

if 'internal' in rootdir:
    center_order = ['BORAMAE', 'KHMC', 'SNUBH', 'GSH']
else:
    center_order = ['MSKCC', 'NCI Urologic Oncology Branch', 'MD Anderson Cancer Center']
    # center_order = ['MSKCC', 'NCI', 'MDACC']
subtype_order = ['KIRC', 'KIRP', 'KICH']


feature_origin_list = ['virchow', 'virchow2', 'UNI', 'UNI2', 'GigaPath', 'CONCH'] 
analysis_target_list = ['subtype', 'center']

        
## extract wd
if True:     
    def compute_psi_from_long(
        dist_long_df: pd.DataFrame,
        sample_col: str = "Sample",
        label_col: str = "Center",     # or "Subtype"
        proto_col: str = "CENTER_proto",  # or "SUBTYPE_proto"
        dist_col: str = "distance",
        trim_q: float = None,   # e.g. 0.95 for tail trimming
    ):
        """
        Compute PSI (mean Wasserstein distance) from long-format distance CSV.
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
                out.append({"Sample": s, f"PSI_{label_col}": np.nan})
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
                    w_dict[f"PSI_{p}"] = 0
                    continue
                d_other = sdf[sdf[proto_col] == p][dist_col].values
                if trim_q is not None:
                    hi = np.quantile(d_other, trim_q)
                    d_other = d_other[d_other <= hi]
                if len(d_true) == 0 or len(d_other) == 0:
                    continue
                wd = wasserstein_distance(d_true, d_other)
                wds.append(wd)
                w_dict[f"PSI_{p}"] = wd

            psi = float(np.mean(wds)) if len(wds) > 0 else np.nan
            w_dict[f"PSI_{label_col}"] = psi
            out.append(w_dict)

        return pd.DataFrame(out)
    
    mitigation_dir = os.path.join(rootdir, mitigations[0])
    dist_dir = os.path.join(mitigation_dir, 'distance')
    for f in feature_origin_list:
        for t in analysis_target_list:
            target_df = pd.read_csv(os.path.join(dist_dir, '%s_dist_to_%s_prototypes_long.csv' % (f, t)))
            
            wd_df = compute_psi_from_long(target_df, label_col=str.upper(t[0])+t[1:], proto_col=str.upper(t)+'_proto')
            
            wd_df.to_csv(os.path.join(dist_dir, '%s_%s_wd.csv' % (f, t)), index=False)
      
## PSI bar/box plot        
if True:
    savedir = os.path.join(rootdir, 'plot')
    for m in mitigations:
        dist_dir = os.path.join(rootdir, m, 'distance')
    
        for t in analysis_target_list:
            total_df = []
            for f in feature_origin_list:
                df = pd.read_csv(os.path.join(dist_dir, '%s_%s_wd.csv' % (f, t)))
                df['feature'] = f
                total_df.append(df)
                
            total_df = pd.concat(total_df)
            
            plt.figure()
            sns.barplot(data=total_df, x='feature', y='PSI_%s' % (str.upper(t[0])+t[1:]), order=feature_origin_list)
            plt.ylim(0, 0.2)
            plt.savefig(os.path.join(savedir, '%s_psi_%s_barplot.png' % (m, t)))
            plt.savefig(os.path.join(savedir, '%s_psi_%s_barplot.svg' % (m, t)), dpi=1000)
            
## PSI bar/box plot per subtype       
if True:
    savedir = os.path.join(rootdir, 'plot')
    for m in mitigations[:1]:
        dist_dir = os.path.join(rootdir, m, 'distance')
    
        for t in analysis_target_list:
            total_df = []
            for f in feature_origin_list:
                df = pd.read_csv(os.path.join(dist_dir, '%s_%s_wd.csv' % (f, t)))
                df['feature'] = f
                total_df.append(df)
                
            total_df = pd.concat(total_df)
            
            if t == 'center': hue_order = [str.upper(t) + '__' + c for c in center_order]
            else : hue_order = [str.upper(t) + '__' + c for c in subtype_order]
            
            plt.figure()
            sns.barplot(data=total_df, x='feature', y='PSI_%s' % (str.upper(t[0])+t[1:]), hue='label', order=feature_origin_list, hue_order=hue_order)
            plt.ylim(0, 0.4)
            
            plt.savefig(os.path.join(savedir, '%s_psi_%s_barplot_per_type.png' % (m, t)))
            plt.savefig(os.path.join(savedir, '%s_psi_%s_barplot_per_type.svg' % (m, t)), dpi=1000)

if True:
    target = 'center'
    label_col = str.upper(target[0])+target[1:]
    psi_col = 'PSI_%s' % label_col
    savedir = os.path.join(rootdir, 'plot',)
    if os.path.exists(savedir) is False: os.mkdir(savedir)
    
    for m in mitigations[1:2]:
        feature_results = []

        for f in feature_origin_list:

            base_df = pd.read_csv(
                os.path.join(rootdir, mitigations[0], 'distance', f'{f}_center_wd.csv'),
                usecols=['Sample', 'label', psi_col]
            )
            xmin, xmax = (min(base_df[psi_col]), max(base_df[psi_col]))
            df = pd.read_csv(
                os.path.join(rootdir, m, 'distance', f'{f}_center_wd.csv'),
                usecols=['Sample', 'label', psi_col]
            )
            subtype_df = pd.read_csv(
                os.path.join(rootdir, m, 'distance', f'{f}_subtype_wd.csv'),
                usecols=['Sample', 'label']
            )

            df['psi_norm'] = (
                df[psi_col]
                .transform(lambda x: (x - xmin) / (xmax - xmin))
            )

            df['subtype'] = [subtype_df[subtype_df['Sample']==s]['label'].item().split('__')[-1] for s in df['Sample']]
            df['center'] = [df.iloc[i]['label'].split('__')[-1] for i in range(len(df))]
            df['feature'] = f
  
            feature_results.append(df)
        total_df = pd.concat(feature_results)

        total_df['center'] = pd.Categorical(
            total_df['center'],
            categories=center_order,
            ordered=True
        )

        total_df['subtype'] = pd.Categorical(
            total_df['subtype'],
            categories=subtype_order,
            ordered=True
        )

        total_df = total_df.sort_values(['center', 'subtype'])
        sample_order = total_df[total_df['feature']==feature_origin_list[0]]['Sample'].tolist()
        total_df.to_csv(os.path.join(savedir, '%s_samplewise_SDD_%s_total.csv' % (m, target)), index=False)
        pivot = total_df[['Sample', 'feature', 'psi_norm']].pivot(index='feature', columns='Sample', values='psi_norm')
        pivot = pivot.loc[feature_origin_list, sample_order]
        pivot.to_csv(os.path.join(savedir, '%s_samplewise_SDD_%s.csv' % (m, target)))
        plt.figure()
        sns.heatmap(pivot, cmap='Reds')
        plt.savefig(os.path.join(savedir, '%s_samplewise_SDD_mean_%s_heatmap.png' % (m, target)))
        plt.savefig(os.path.join(savedir, '%s_samplewise_SDD_mean_%s_heatmap.svg' % (m, target)), dpi=1000)
    