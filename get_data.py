import os
from datetime import datetime
import pickle as pk
from convokit import download
from preprocess import preprocess
import jsonlines
from tqdm import tqdm
import numpy as np
import argparse

data_dir = "./data/"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subreddit", required=True, type=str)
args = parser.parse_args()

print(args.subreddit)
download(f'subreddit-{args.subreddit}', data_dir='./data/convokit_downloads/')

comments = list()
with jsonlines.open(f'./data/convokit_downloads/subreddit-{args.subreddit}/utterances.jsonl') as reader:
    for obj in tqdm(reader):
        if obj['timestamp'] > 1388552400 and obj['reply_to']: # only collect comments after 2014/1
            if len(obj['text'].split()) > 5:
                obj['text'] = preprocess(obj['text'])
                if len(obj['text']) > 5:
                    dt = datetime.fromtimestamp(obj['timestamp'])
                    obj['year'] = dt.year
                    obj['month'] = dt.month
                    obj['year-month'] = str(dt.year) + "-" + str(dt.month).zfill(2)
                    comments.append({k: obj[k] for k in ['id', 'year', 'month',
                                                         'timestamp', 'year-month', 'text']})

os.system(f"rm -r ./data/convokit_downloads/*{args.subreddit}*")
sample_comments = np.random.choice(comments, size=min(len(comments), 500000), replace=False)

pk.dump(sample_comments, open(f"./data/samples/{args.subreddit}-comments.pk", "wb"))
