from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


def dist_for_one_sample(metadata, embeddings):
    wk2mn = dict()

    sample_idx = np.random.choice(len(embeddings), size=50000, replace=True)
    sample_meta = metadata[sample_idx]
    sample_emb = embeddings[sample_idx, :]
    for r in sample_meta:
        isocal = datetime.fromtimestamp(r['timestamp']).isocalendar()
        r['year-week'] = f"{isocal.year}-{str(isocal.week).zfill(2)}"

        mn = wk2mn.get(r['year-week'])
        if mn is not None:
            wk2mn[r['year-week']] = min(mn, r['year-month'])
        else:
            wk2mn[r['year-week']] = r['year-month']

    weekly_emb = pd.DataFrame(sample_emb).groupby([x['year-week'] for x in sample_meta]).mean()
    dist = euclidean_distances(weekly_emb.values)
    np.fill_diagonal(dist, np.nan)
    dist_df = pd.DataFrame(dist, index=weekly_emb.index, columns=weekly_emb.index)

    long = dist_df.unstack().reset_index()
    long = long.sort_values(["level_0", "level_1"])
    long = long.rename({"level_0": "week_1", "level_1": "week_2", 0: "emb"}, axis=1)

    long['month_1'] = long['week_1'].apply(lambda w: wk2mn[w])
    long['month_2'] = long['week_2'].apply(lambda w: wk2mn[w])
    return long


def run_regression(long):
    long["n_months"] = np.abs(
        (pd.to_datetime(long["month_2"]).dt.to_period("M") - pd.to_datetime(long["month_1"]).dt.to_period(
            "M")).apply(
            lambda x: x.n))
    long = long[long['n_months'] <= 36].copy()
    long = long.fillna(0)
    long['same_year'] = long.apply(lambda r: 1 if r['month_1'].split("-")[0] == r['month_2'].split("-")[0] else 0,
                                   axis=1)
    long['same_month'] = long.apply(lambda r: 1 if r['month_1'].split("-")[1] == r['month_2'].split("-")[1] else 0,
                                    axis=1)

    scaler = preprocessing.MinMaxScaler()
    norms = scaler.fit_transform(long[['emb', "n_months"]].values)
    long['normalized_emb'] = norms[:, 0]
    long['normalized_n_months'] = norms[:, 1]
    # drop duplicates
    x = long.dropna().drop_duplicates(subset=["emb", "n_months", "same_year", "same_month"])
    mod = smf.ols(formula='normalized_emb ~ C(same_year) + C(same_month) + normalized_n_months', data=x)
    dist_res = mod.fit()

    return dist_res


def bootstrap(metadata, embeddings, n=500):
    dist_params = list()

    for i in tqdm(range(n)):
        long = dist_for_one_sample(metadata, embeddings)
        dist_res = run_regression(long)

        for k, v in dist_res.params.to_dict().items():
            dist_params.append({"variable": k, "coefficient": v, "pvalue": dist_res.pvalues[k]})
    return dist_params


if __name__ == '__main__':
    import argparse
    import pickle as pk

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)
    subreddit = args.subreddit
    metadata = pk.load(open(f"./data/samples/{subreddit}-comments.pk", "rb"))
    embeddings = pk.load(open(f"./data/output/embeddings/{subreddit}.pk", "rb"))

    dist_params = bootstrap(metadata, embeddings, 500)
    pk.dump(dist_params, open(f"data/output/regression/{subreddit}-emb_params.pk", "wb"))
