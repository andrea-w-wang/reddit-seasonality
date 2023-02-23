import argparse
from collections import defaultdict
from itertools import combinations
import pickle as pk
import metrics

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subreddit", required=True, type=str)
args = parser.parse_args()

print(args.subreddit)

ngrams_counter = pk.load(open(f"data/output/ngrams/{args.subreddit}.pk", "rb"))
all_months = sorted(ngrams_counter.keys())
month_pairs = list(combinations(all_months, r=2))

jsd_diffs = defaultdict(dict)
for month_1, month_2 in month_pairs:
    print("\t", month_1, month_2)

    my_jsd = metrics.JSD(ngrams_counter[month_1],
                         ngrams_counter[month_2],
                         weight_1=0.5, weight_2=0.5,
                         base=2)

    jsd_diffs[month_1][month_2] = my_jsd.total_diff
