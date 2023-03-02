import pickle as pk

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from tqdm import tqdm

import jsd
import ngrams


def dist_for_one_sample(metadata):
    sample_idx = np.random.choice(len(metadata), size=50000, replace=True)
    sample_meta = metadata[sample_idx]
    ngrams_counter = ngrams.get_ngrams_counter(utts=sample_meta)
    months, dist, long = jsd.run(ngrams_counter)
    return dist, months, long


def run_regression(long):
    long["n_months"] = np.abs(
        (pd.to_datetime(long["month_2"]).dt.to_period("M") - pd.to_datetime(long["month_1"]).dt.to_period("M")).apply(
            lambda x: x.n))
    long = long[long['n_months'] <= 36].copy()
    long['month_1_year'] = long['month_1'].apply(lambda t: t.split("-")[0])
    long['month_1_month'] = long['month_1'].apply(lambda t: t.split("-")[1])
    long['same_year'] = long.apply(lambda r: 1 if r['month_1'].split("-")[0] == r['month_2'].split("-")[0] else 0,
                                   axis=1)
    long['same_month'] = long.apply(lambda r: 1 if r['month_1'].split("-")[1] == r['month_2'].split("-")[1] else 0,
                                    axis=1)
    x = long.dropna().drop_duplicates(subset=["jsd", "same_year", "same_month", "month_1_year", "month_1_month"])
    mod = smf.ols(formula='jsd ~ C(same_year) + C(same_month) + C(month_1_year) + C(month_1_month)', data=x)
    dist_res = mod.fit()

    mod = OrderedModel.from_formula("jsd_rank ~ C(same_year) + C(same_month) + C(month_1_year) + C(month_1_month)",
                                    data=long.dropna(),
                                    distr='logit')
    rank_res = mod.fit(method='bfgs')
    return dist_res, rank_res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)
    metadata = pk.load(open(f"./data/samples/{args.subreddit}-comments.pk", "rb"))

    dist_params = list()
    rank_params = list()
    dist_stash = None
    for i in tqdm(range(500)):
        dist, months, long = dist_for_one_sample(metadata)
        dist_res, rank_res = run_regression(long)

        if i == 0:
            dist_stash = np.array([dist])
        else:
            dist_stash = np.concatenate((dist_stash, np.array([dist])))

        for k, v in dist_res.params.to_dict().items():
            dist_params.append({"variable": k, "coefficient": v, "pvalue": dist_res.pvalues[k]})

        for k, v in rank_res.params.to_dict().items():
            rank_params.append({"variable": k, "coefficient": v, "pvalue": rank_res.pvalues[k]})

    dist_stash[np.isnan(dist_stash)] = 0

    pk.dump((months, dist_stash), open(f"data/output/distances/{args.subreddit}-jsd-II.pk", "wb"))
    pk.dump(rank_params, open(f"data/output/regression/{args.subreddit}-jsd_rank_params-II.pk", "wb"))
    pk.dump(dist_params, open(f"data/output/regression/{args.subreddit}-jsd_params-II.pk", "wb"))

    # plot distance heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"r/{args.subreddit} Jensen-Shannon Divergence")
    mean_stash = dist_stash.mean(axis=0)
    mean_stash[mean_stash == 0] = np.nan
    dist = pd.DataFrame(mean_stash, index=months)
    dist.columns = months
    ax = sns.heatmap(dist, cmap='PiYG', ax=axes[0])
    ax.set_title(f"JSD")

    from scipy.stats import rankdata
    mean_rank = rankdata(dist_stash, axis=1).mean(axis=0)
    np.fill_diagonal(mean_rank, np.nan)
    dist = pd.DataFrame(mean_rank, index=months)
    dist.columns = months
    sns.heatmap(dist, cmap='PiYG', ax=axes[1])
    axes[1].set_title(f"Rank by JSD")
    plt.savefig(f"./figures/{args.subreddit}-jsd-heatmap-II.jpg", bbox_inches='tight')

    # plot dist params
    df = pd.DataFrame(dist_params)

    for variable_name in ['same_month', 'same_year', 'month_1_year', 'month_1_month']:
        cat_df = df[df['variable'].str.contains(variable_name)].copy()
        cat_df['variable_value'] = cat_df['variable'].apply(lambda v: v.split("T.")[-1].strip("]"))
        cat_df = cat_df.rename({"variable": variable_name}, axis=1)

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f"r/{args.subreddit} regression, output = JSD")
        ax = sns.boxplot(data=cat_df, x='variable_value', y='coefficient', ax=axes[0])
        ax.set_xlabel(variable_name)
        ax.set_title(f"coefficients")

        ax = sns.boxplot(data=cat_df, x='variable_value', y='pvalue', ax=axes[1])
        ax.set_xlabel(variable_name)
        ax.set_title(f"p values")
        ax.axhline(0.05, color='red', label='p = 0.05')
        plt.legend()
        plt.savefig(f"./figures/{args.subreddit}-jsd-regression-{variable_name}-II.jpg", bbox_inches='tight')

    # plot rank params
    df = pd.DataFrame(rank_params)
    for variable_name in ['same_month', 'same_year', 'month_1_year', 'month_1_month']:
        cat_df = df[df['variable'].str.contains(variable_name)].copy()
        cat_df['variable_value'] = cat_df['variable'].apply(lambda v: v.split("T.")[-1].strip("]"))
        cat_df = cat_df.rename({"variable": variable_name}, axis=1)

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f"r/{args.subreddit} regression, output = Rank by JSD")
        ax = sns.boxplot(data=cat_df, x='variable_value', y='coefficient', ax=axes[0])
        ax.set_xlabel(variable_name)
        ax.set_title(f"coefficients")

        ax = sns.boxplot(data=cat_df, x='variable_value', y='pvalue', ax=axes[1])
        ax.set_xlabel(variable_name)
        ax.set_title(f"p values")
        ax.axhline(0.05, color='red', label='p = 0.05')
        plt.legend()
        plt.savefig(f"./figures/{args.subreddit}-jsd-rank-regression-{variable_name}-II.jpg", bbox_inches='tight')