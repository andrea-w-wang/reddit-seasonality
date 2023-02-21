import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import pickle as pk
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--perplexity", required=True, type=float)
args = parser.parse_args()

combined_emb_df = pk.load(open("./data/combined_emb_df.pk", "rb"))
embeddings = combined_emb_df.drop(['id', 'year-month', 'file'], axis=1).values
method = TSNE(n_components=2, perplexity=args.perplexity)
projection = method.fit_transform(embeddings)
metadata = combined_emb_df[['id', 'year-month', 'file']].reset_index(drop=True)
metadata['subreddit'] = metadata['file'].apply(lambda f: f.split("-")[0])
xdf = pd.concat((pd.DataFrame(projection), metadata), axis=1)
sns.scatterplot(data=xdf, x=0, y=1, hue='subreddit')
plt.title(f"TSNE-perplexity_{args.perplexity}")
plt.savefig(f"./figures/TSNE-perplexity_{args.perplexity}.png")


