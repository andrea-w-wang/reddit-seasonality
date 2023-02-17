import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns
# initialize ST5 model
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run(subreddit):
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    df = pk.load(open(f"data/{subreddit}-comments.pk", "rb"))
    sentences = list(df['text'].values)

    embeddings = model.encode(sentences)
    return embeddings


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str, nargs='+')
    args = parser.parse_args()

    for subreddit in ars.subreddit:
        embeddings = run(args.subreddit)
        pk.dump(random_samples, open(f"./data/{args.subreddit}-sentence-embeddings.pk", "wb"))
