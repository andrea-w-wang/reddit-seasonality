import pickle as pk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances
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


def plot_distance_heatmap(months, dist_stash, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"r/{args.subreddit} Sentence Embedding")
    mean_stash = dist_stash.mean(axis=0)
    mean_stash[mean_stash == 0] = np.nan
    dist = pd.DataFrame(mean_stash, index=months)
    dist.columns = months
    sns.heatmap(dist, cmap='PiYG', ax=axes[0])
    axes[0].set_title(f"Euclidean distance")

    mean_rank = rankdata(dist_stash, axis=1).mean(axis=0)
    np.fill_diagonal(mean_rank, np.nan)
    dist = pd.DataFrame(mean_rank, index=months)
    dist.columns = months
    sns.heatmap(dist, cmap='PiYG', ax=axes[1])
    axes[1].set_title(f"Rank distance")
    plt.savefig(output_path, bbox_inches='tight')


def bootstrap(run_regression, n=500):
    dist_params = list()
    rank_params = list()
    dist_stash = None
    months = None
    for i in tqdm(range(n)):
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

    return months, dist_stash, dist_params, rank_params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)
    metadata = pk.load(open(f"./data/samples/{args.subreddit}-comments.pk", "rb"))
    embeddings = pk.load(open(f"./data/output/embeddings/{args.subreddit}.pk", "rb"))

    months, dist_stash, dist_params, rank_params = bootstrap(n=500)

    pk.dump((months, dist_stash), open(f"data/output/distances/{args.subreddit}-emb.pk", "wb"))
    pk.dump(rank_params, open(f"data/output/regression/{args.subreddit}-emb_rank_params.pk", "wb"))
    pk.dump(dist_params, open(f"data/output/regression/{args.subreddit}-emb_params.pk", "wb"))

    plot_distance_heatmap(months, dist_stash, f"./figures/{args.subreddit}-emb-heatmap.jpg")

    # plot dist params
    import re
    df = pd.DataFrame(dist_params)
    cat_df = df[df['variable'].str.startswith("C(")].copy()
    cat_df['variable_value'] = cat_df['variable'].apply(lambda v: v.split("T.")[-1].strip("]"))
    cat_df['variable_name'] = cat_df['variable'].apply(lambda v: re.match("C\((.*)\)", v).group(1))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f"r/{args.subreddit} regression, output = embedding distance")
    ax = sns.boxplot(data=cat_df, x='variable_value', y='coefficient', ax=axes[0])
    ax.set_xlabel("n_months")
    ax.set_title(f"coefficients")

    ax = sns.boxplot(data=cat_df, x='variable_value', y='pvalue', ax=axes[1])
    ax.set_xlabel("n_months")
    ax.set_title(f"p values")
    ax.axhline(0.05, color='red', label='p = 0.05')
    plt.legend()
    plt.savefig(f"./figures/{args.subreddit}-emb-regression.jpg", bbox_inches='tight')

    # plot rank params
    df = pd.DataFrame(rank_params)
    cat_df = df[df['variable'].str.startswith("C(")].copy()
    cat_df['variable_value'] = cat_df['variable'].apply(lambda v: v.split("T.")[-1].strip("]"))
    cat_df['variable_name'] = cat_df['variable'].apply(lambda v: re.match("C\((.*)\)", v).group(1))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f"r/{args.subreddit} regression, output = embedding distance rank")
    ax = sns.boxplot(data=cat_df, x='variable_value', y='coefficient', ax=axes[0])
    ax.set_xlabel("n_months")
    ax.set_title(f"coefficients")

    ax = sns.boxplot(data=cat_df, x='variable_value', y='pvalue', ax=axes[1])
    ax.set_xlabel("n_months")
    ax.set_title(f"p values")
    ax.axhline(0.05, color='red', label='p = 0.05')
    plt.legend()
    plt.savefig(f"./figures/{args.subreddit}-emb-rank-regression.jpg", bbox_inches='tight')
