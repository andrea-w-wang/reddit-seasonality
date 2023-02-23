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

print(args.subreddit)
sample_df_folder = './data/samples/'
data = pk.load(open(f"{sample_df_folder}{args.subreddit}-comments.pk", "rb"))
model = SentenceTransformer('sentence-transformers/sentence-t5-base', device=device)

sentences = [x['text'] for x in data]
embeddings = model.encode(sentences)
pk.dump(embeddings, open(f"./data/output/embeddings/{args.subreddit}.pk", "wb"))
