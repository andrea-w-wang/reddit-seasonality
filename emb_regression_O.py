import pickle as pk

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.miscmodels.ordinal_model import OrderedModel
from tqdm import tqdm


def dist_for_one_sample(metadata, embeddings):
    sample_idx = np.random.choice(len(embeddings), size=50000, replace=True)
    sample_meta = metadata[sample_idx]
    sample_emb = embeddings[sample_idx, :]
    monthly_emb = pd.DataFrame(sample_emb).groupby([x['year-month'] for x in sample_meta]).mean()
    dist = euclidean_distances(monthly_emb.values)
    months = list(monthly_emb.index)
    dist_df = pd.DataFrame(dist, index=months)
    dist_df.columns = months
    long = dist_df.unstack().reset_index()
    long = long.sort_values(["level_0", "level_1"])
    long = long.rename({"level_0": "month_1", "level_1": "month_2", 0: "emb"}, axis=1)
    long['emb_rank'] = long.groupby("month_1")["emb"].rank()

    return dist, months, long


def run_regression(long):
    long["n_months"] = np.abs(
        (pd.to_datetime(long["month_2"]).dt.to_period("M") - pd.to_datetime(long["month_1"]).dt.to_period("M")).apply(
            lambda x: x.n))
    long = long[long['n_months'] <= 36].copy()

    long['same_year'] = long.apply(lambda r: 1 if r['month_1'].split("-")[0] == r['month_2'].split("-")[0] else 0,
                                   axis=1)
    long['same_month'] = long.apply(lambda r: 1 if r['month_1'].split("-")[1] == r['month_2'].split("-")[1] else 0,
                                    axis=1)

    x = long.dropna().drop_duplicates(subset=["emb", "same_year", "same_month", "n_months"])
    mod = smf.ols(formula='emb ~ C(same_year) + C(same_month) + n_months', data=x)
    dist_res = mod.fit()

    mod = OrderedModel.from_formula("emb_rank ~ C(same_year) + C(same_month) + n_months", data=long.dropna(),
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
    embeddings = pk.load(open(f"./data/output/embeddings/{args.subreddit}.pk", "rb"))

    dist_params = list()
    rank_params = list()
    dist_stash = None
    for i in tqdm(range(500)):
        dist, months, long = dist_for_one_sample(metadata, embeddings)
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

    pk.dump((months, dist_stash), open(f"data/output/distances/{args.subreddit}-emb-O.pk", "wb"))
    pk.dump(rank_params, open(f"data/output/regression/{args.subreddit}-emb_rank_params-O.pk", "wb"))
    pk.dump(dist_params, open(f"data/output/regression/{args.subreddit}-emb_params-O.pk", "wb"))

    # plot distance heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"r/{args.subreddit} Sentence Embedding")
    mean_stash = dist_stash.mean(axis=0)
    mean_stash[mean_stash == 0] = np.nan
    dist = pd.DataFrame(mean_stash, index=months)
    dist.columns = months
    sns.heatmap(dist, cmap='PiYG', ax=axes[0])
    axes[0].set_title(f"Euclidean distance")

    from scipy.stats import rankdata

    mean_rank = rankdata(dist_stash, axis=1).mean(axis=0)
    np.fill_diagonal(mean_rank, np.nan)
    dist = pd.DataFrame(mean_rank, index=months)
    dist.columns = months
    sns.heatmap(dist, cmap='PiYG', ax=axes[1])
    axes[1].set_title(f"Rank")
    plt.savefig(f"./figures/{args.subreddit}-emb-heatmap-O.jpg", bbox_inches='tight')

    # plot dist params
    df = pd.DataFrame(dist_params)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f"r/{args.subreddit} regression, output = embedding distance")
    ax = sns.boxplot(data=df, x='variable', y='coefficient', ax=axes[0])
    ax.set_title("coefficients")

    ax = sns.boxplot(data=df, x='variable', y='pvalue', ax=axes[1])
    ax.set_title(f"p values")
    ax.axhline(0.05, color='red', label='p = 0.05')
    plt.legend()
    plt.savefig(f"./figures/{args.subreddit}-emb-regression-O.jpg", bbox_inches='tight')

    # plot rank params
    df = pd.DataFrame(rank_params)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f"r/{args.subreddit} regression, output = Rank by embedding distance")
    ax = sns.boxplot(data=df, x='variable', y='coefficient', ax=axes[0])
    ax.set_title("coefficients")

    ax = sns.boxplot(data=df, x='variable', y='pvalue', ax=axes[1])
    ax.set_title(f"p values")
    ax.axhline(0.05, color='red', label='p = 0.05')
    plt.legend()
    plt.savefig(f"./figures/{args.subreddit}-emb-rank-regression-O.jpg", bbox_inches='tight')