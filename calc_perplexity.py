import argparse
import pickle as pk
import torch
from datasets import load_dataset, Dataset
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from scipy import stats
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from evaluate import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "./data/samples"


def run(subreddit, sample_filename, model_month, model_name="distilgpt2"):
    """

    :return: [{"model_month", "predict_month", "sample_file", "output"}]
    """
    random_samples = pk.load(open(f"{data_dir}/{subreddit}-{sample_filename}.pk", "rb"))
    ppl_results = list()
    myppl = myPerplexity()
    for predict_month in random_samples['year-month'].unique():
        selected_random_utts = list(random_samples[random_samples['year-month'] == predict_month]['text'].values)

        stash = dict()
        stash['model_month'] = model_month
        stash['predict_month'] = predict_month
        stash['sample_filename'] = sample_filename
        print(model_month)
        checkpoint_path = f"./models/{model_name}_{subreddit}_{model_month}/best"
        stash.update(myppl._compute(
            data=selected_random_utts,
            model_id=checkpoint_path,
            add_start_token=False,  # default,
            max_length=1024
        ))

        ppl_results.append(stash)
    return ppl_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str)
    parser.add_argument("-f", "--sample-filename", required=True, type=str)
    parser.add_argument("-month", "--model-month", required=True, type=str)
    parser.add_argument("-m", "--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()
    output = run(args.subreddit, args.sample_filename, args.model_month, args.model_name)
    pk.dump(output, open(
        f"./data/output/{args.subreddit}-{args.sample_filename}-model_month{args.model_month}.pk", "wb"
    ))

