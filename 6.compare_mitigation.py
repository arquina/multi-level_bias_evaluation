
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
import argparse

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
        
def mitigation_comparison(root_dir, pfm_list,save_dir, target_column, target_mitigation):
    mitigation_basedir = os.path.join(root_dir, 'original')
    save_dir = os.path.join(save_dir, target_mitigation)
    os.makedirs(save_dir, exist_ok= True)

    pdd_col = 'PDD_%s' % target_column
    maxy = [0.44, 0.11, 0.22, 0.22, 0.165, 0.11]
    maxy = [0.22, 0.088, 0.154, 0.154, 0.154, 0.088]
    
    mitigation_dir = os.path.join(root_dir, target_mitigation)

    fig, ax = plt.subplots(1, len(pfm_list), figsize=(30, 5))
    for i, pfm in enumerate(pfm_list):
        b_df = pd.read_csv(
            os.path.join(mitigation_basedir, 'distance', f'{pfm}_{target_column}_psi.csv'),
            usecols=['sample', pdd_col]
        )
        m_df = pd.read_csv(
            os.path.join(mitigation_dir, 'distance', f'{pfm}_{target_column}_psi.csv'),
            usecols=['sample', pdd_col]
        )

        b_df['mitigation'] = 'original'
        m_df['mitigation'] = target_mitigation
        total_df = pd.concat([b_df, m_df])
        total_df = total_df.groupby(['mitigation', 'sample'])[pdd_col].mean().reset_index()
        print(total_df)
        test, pvalue = select_significance(b_df[pdd_col], m_df[pdd_col], paired=True)
        print(test, pvalue)
        # boxplot
        sns.boxplot(
            data=total_df,
            x="mitigation",
            y=pdd_col,
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
            y=pdd_col,
            # hue="Sample",
            dodge=False,
            jitter=False,
            size=4,
            alpha=0.6,
            legend=False,
            ax=ax[i]
            
        )

        # connect paired points
        for sid, sub_df in total_df.groupby('sample'):
            ax[i].plot(
                [0, 1],
                sub_df[pdd_col],
                color="gray",
                alpha=0.4,
                linewidth=1
            )
        ax[i].set_ylim(0, maxy[i])
        ax[i].set_title('%s_%s (%s: %.2e)' % (target_column, pfm, test, pvalue))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '%s_pdd_%s_paired_boxplot.png' % (target_column, target_mitigation)))             
    plt.savefig(os.path.join(save_dir, '%s_pdd_%s_paired_boxplot.svg' % (target_column, target_mitigation)), dpi=1000)    


def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--root_dir", help = 'root_dir of analysis', type = str, required = False)
    parser.add_argument("--metadata", help = 'Path of the metadata', type = str, required = False)
    parser.add_argument("--save_dir", help = 'Directory to save the feature',type = str, required = False)
    parser.add_argument("--pfm_list", nargs = "+", default = [], help = 'PFM list for comparison', type = str)
    parser.add_argument("--target_column", default = 'center', help = 'Cateogry to make prototype (e.g. subtype, center, scanner, race)')
    parser.add_argument("--target_mitigation", default = 'stainnorm', help = 'Mitigation for compare', type = bool)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    mitigation_comparison(Argument.root_dir, Argument.pfm_list, Argument.save_dir, Argument.target_column, Argument.target_mitigation)
        
if __name__ == "__main__":
    main()