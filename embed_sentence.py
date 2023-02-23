import pickle as pk

import numpy as np
import pandas as pd
# initialize ST5 model
from sentence_transformers import SentenceTransformer
import argparse
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subreddit", required=True, type=str)
args = parser.parse_args()

sample_df_folder = './data/samples/'
data = pk.load(open(f"{sample_df_folder}{args.subreddit}-comments.pk", "rb"))
model = SentenceTransformer('sentence-transformers/sentence-t5-base', device=device)

for i in range(10):
    print(f"sample {i+1}")
    sample = np.random.choice(data, size=50000, replace=False)
    sentences = [x['text'] for x in sample]
    embeddings = model.encode(sentences)
    monthly_embed = pd.DataFrame(embeddings).groupby([x['year-month'] for x in sample]).mean()
    stash = (np.array(monthly_embed.index), monthly_embed.values)

    pk.dump(stash, open(f"./data/output/embeddings/{args.subreddit}-sample{i+1}.pk", "wb"))
