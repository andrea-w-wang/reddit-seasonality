import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns
import umap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_neighbors", required=True, type=float)
parser.add_argument("-d", "--min_dist", required=True, type=float)
args = parser.parse_args()

combined_emb_df = pk.load(open("./data/combined_emb_df.pk", "rb"))
embeddings = combined_emb_df.drop(['id', 'year-month', 'file'], axis=1).values

method = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric='euclidean'
    )
projection = method.fit_transform(embeddings)
metadata = combined_emb_df[['id', 'year-month', 'file']].reset_index(drop=True)
metadata['subreddit'] = metadata['file'].apply(lambda f: f.split("-")[0])
xdf = pd.concat((pd.DataFrame(projection), metadata), axis=1)
sns.scatterplot(data=xdf, x=0, y=1, hue='subreddit', alpha=0.5)
plt.title(f"UMAP-n_neighbors={args.n_neighbors}-min_dist={args.min_dist}")
plt.savefig(f"./figures/UMAP-neighbors_{args.n_neighbors}-mindist_{args.min_dist}.png")
