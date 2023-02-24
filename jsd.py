import argparse
import pickle as pk
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

import metrics


def run(ngrams_counter):
    all_months = sorted(ngrams_counter.keys())
    month_pairs = list(combinations(all_months, r=2))

    jsd_diffs = defaultdict(dict)
    for month_1, month_2 in month_pairs:

        my_jsd = metrics.JSD(ngrams_counter[month_1],
                             ngrams_counter[month_2],
                             weight_1=0.5, weight_2=0.5,
                             base=2)

        jsd_diffs[month_1][month_2] = my_jsd.total_diff

    first_month = all_months[0]
    last_month = all_months[-1]
    jsd_diffs[first_month][first_month] = np.nan
    jsd_diffs[last_month][last_month] = np.nan

    jsd_df = pd.DataFrame(jsd_diffs)
    jsd_df = jsd_df.sort_index()
    X = jsd_df.fillna(0).values
    X2 = X + X.T - np.diag(np.diag(X))
    jsd_df = pd.DataFrame(X2, columns=jsd_df.columns, index=jsd_df.index)
    jsd_df = jsd_df.replace(0, np.nan)

    long = jsd_df.unstack().reset_index()
    long = long.sort_values(["level_0", "level_1"])
    long = long.rename({"level_0": "month_1", "level_1": "month_2", 0: "jsd"}, axis=1)
    long['jsd_rank'] = long.groupby("month_1")["jsd"].rank()
    return long


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)

    ngrams_counter = pk.load(open(f"data/output/ngrams/{args.subreddit}.pk", "rb"))
    long = run(ngrams_counter)
    pk.dump(long, open(f"data/output/long/jsd/{args.subreddit}.pk", "wb"))
