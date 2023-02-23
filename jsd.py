import collections.abc
import pickle as pk
from collections import Counter

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

collections.Mapping = collections.abc.Mapping

data_dir = "./data/"


def count_ngrams(input_texts, n=1):
    ngrams_counter = Counter()
    for text in input_texts:
        text_ngrams = ngrams(word_tokenize(text), n)
        ngrams_counter.update(text_ngrams)
    return ngrams_counter


def get_ngrams_counter(utt_fp):
    ngrams_counter = dict()

    data = pk.load(open(utt_fp, "rb"))
    df = pd.DataFrame(list(data))
    months = df['year-month'].unique()
    for m in months:
        print("\t", m)
        monthly_comments = df[df['year-month'] == m]['text'].tolist()
        ngrams_counter[m] = count_ngrams(monthly_comments)

    return ngrams_counter


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", required=True, type=str)
    args = parser.parse_args()

    print(args.subreddit)

    sample_df_folder = './data/samples/'
    utt_fp = f"{sample_df_folder}{args.subreddit}-comments.pk"
    stash = get_ngrams_counter(utt_fp)

    pk.dump(stash, open(f"{args.subreddit}.pk", "wb"))

