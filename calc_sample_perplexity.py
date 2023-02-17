import os
import evaluate
import numpy as np
import pickle as pk
import torch
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

import metrics

data_dir = "./data/"
device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(0)


def calculate_my_ppl(checkpoint_path, utterances):
    perplexity = metrics.myPerplexity()
    ppl = perplexity._compute(
        model_id=checkpoint_path,
        add_start_token=False,
        data=utterances,
        max_length=1024
    )
    return ppl


def run(subreddit, utterance_filepath, model_month, model_name="distilgpt2"):
    utts = pk.load(open(utterance_filepath, 'rb'))
    utts = list(utts['text'].values)
    print(len(utts))
    checkpoint_path = f"./models/{model_name}_{subreddit}_{model_month}/best"
    print(checkpoint_path)
    output = calculate_my_ppl(checkpoint_path, utts)

    utterance_filename = os.path.basename(utterance_filepath).split(".")[0]
    pk.dump(output,
            open(data_dir + f"output/{model_name}_{subreddit}_{model_month}-{utterance_filename}-scores.pk", "wb"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--subreddit", required=True, type=str)
    parser.add_argument("-fp", "--utterance-filepath", required=True, type=str)
    parser.add_argument("-month", "--model-month", required=True, type=str)
    parser.add_argument("-model", "--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()
    run(args.subreddit, args.utterance_filepath, args.model_month, args.model_name)
