import pickle as pk
import statistics
from collections import defaultdict
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dateutil import relativedelta
from dateutil.parser import parse
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.miscmodels.ordinal_model import OrderedModel
from tqdm import tqdm


def run_one_sample(metadata, embeddings):
    sample_idx = np.random.choice(len(embeddings), size=50000, replace=True)
    sample_meta = metadata[sample_idx]
    sample_emb = embeddings[sample_idx, :]
    monthly_emb = pd.DataFrame(sample_emb).groupby([x['year-month'] for x in sample_meta]).mean()
    dist = euclidean_distances(monthly_emb.values)
    np.fill_diagonal(dist, np.nan)
    months = list(monthly_emb.index)
    dist_df = pd.DataFrame(dist, index=months)
    dist_df.columns = months
    long = dist_df.unstack().reset_index()
    long = long.sort_values(["level_0", "level_1"])
    long = long.rename({"level_0": "month_1", "level_1": "month_2", 0: "emb"}, axis=1)
    long['emb_rank'] = long.groupby("month_1")["emb"].rank()
    long['same_year'] = long.apply(lambda r: 1 if r['month_1'].split("-")[0] == r['month_2'].split("-")[0] else 0,
                                   axis=1)
    long['same_month'] = long.apply(lambda r: 1 if r['month_1'].split("-")[1] == r['month_2'].split("-")[1] else 0,
                                    axis=1)
    long['months_apart'] = long.apply(lambda r:
                                      np.abs(
                                          relativedelta.relativedelta(parse(r['month_2']), parse(r['month_1'])).months),
                                      axis=1)

    scaler = preprocessing.MinMaxScaler()
    norms = scaler.fit_transform(long[['emb', 'emb_rank', "months_apart"]].values)
    long['normalized_emb'] = norms[:, 0]
    long['normalized_emb_rank'] = norms[:, 1]
    long['normalized_months_apart'] = norms[:, 2]
    mod = smf.ols(formula='normalized_emb ~ C(same_year) + C(same_month) + normalized_months_apart', data=long.dropna())
    dist_res = mod.fit()

    mod = OrderedModel.from_formula("emb_rank ~ C(same_year) + C(same_month) + months_apart", data=long.dropna(),
                                    distr='logit')

    rank_res = mod.fit(method='bfgs')

    return dist_res, rank_res


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25, ax=None):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval

    if ax:
        ax.plot([x, x], [top, bottom], color=color)
        ax.plot([left, right], [top, top], color=color)
        ax.plot([left, right], [bottom, bottom], color=color)
        ax.plot(x, mean, 'o', color='#f44336')

    else:
        plt.plot([x, x], [top, bottom], color=color)
        plt.plot([left, right], [top, top], color=color)
        plt.plot([left, right], [bottom, bottom], color=color)
        plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)
    metadata = pk.load(open(f"./data/samples/{args.subreddit}-comments.pk", "rb"))
    embeddings = pk.load(open(f"./data/output/embeddings/{args.subreddit}.pk", "rb"))

    emb_params = defaultdict(list)
    rank_params = defaultdict(list)

    for i in tqdm(range(500)):
        emb_res, rank_res = run_one_sample(metadata, embeddings)

        for k, v in emb_res.params.to_dict().items():
            emb_params[k].append({"param": v, "pvalue": emb_res.pvalues[k]})

        for k, v in rank_res.params.to_dict().items():
            rank_params[k].append({"param": v, "pvalue": rank_res.pvalues[k]})

    pk.dump(rank_params, open(f"data/output/regression/{args.subreddit}-emb_rank_params.pk", "wb"))
    pk.dump(emb_params, open(f"data/output/regression/{args.subreddit}-emb_params.pk", "wb"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    params = list(rank_params.keys())
    axes[0].set_xticks(range(1, len(params) + 1), params)
    axes[0].set_title(f'r/{args.subreddit}, output = embedding distance ranking')
    for i, p in enumerate(params):
        plot_confidence_interval(i + 1, rank_params[p], ax=axes[0])

    params = list(emb_params.keys())
    axes[1].set_xticks(range(1, len(params) + 1), params)
    axes[1].set_title(f'r/{args.subreddit}, output = embedding distance')
    for i, p in enumerate(params):
        plot_confidence_interval(i + 1, emb_params[p], ax=axes[1])
    plt.savefig(f"./figures/{args.subreddit}-emb-reg-minmax.png")
