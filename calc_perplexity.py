import argparse
import evaluate
import pickle as pk
import torch
from datasets import load_dataset, Dataset
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from scipy import stats
# Transformer library
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

import metrics

data_dir = "./data/samples"
device_name = "cuda" if torch.cuda.is_available() else "cpu"


def run(subreddit, sample_filename, start_model_month, end_model_month, model_name="distilgpt2"):
    """

    :return: [{"model_month", "predict_month", "sample_file", "output"}]
    """
    random_samples = pk.load(open(f"{data_dir}/{subreddit}-{sample_filename}.pk", "rb"))
    ppl_results = list()
    myppl = metrics.myPerplexity()
    for predict_month in random_samples['year-month'].unique():
        selected_random_utts = list(random_samples[random_samples['year-month'] == predict_month]['text'].values)

        model_month = start_model_month
        while parse(model_month) <= parse(end_model_month):
            stash = dict()
            stash['predict_month'] = predict_month
            stash['sample_filename'] = sample_filename
            print(model_month)
            checkpoint_path = f"./models/{model_name}_{subreddit}_{model_month}/best"
            stash['model_month'] = model_month
            stash['ppls'] = myppl._compute(
                data=selected_random_utts,
                model_id=checkpoint_path,
                add_start_token=False,  # default,
                max_length=1024
            )

            model_month = (parse(model_month) + relativedelta(months=1)).strftime("%Y-%m")
            ppl_results.append(stash)

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-sr", "--subreddit", required=True, type=str)
#     parser.add_argument("-month", "--model-month", required=True, type=str)
#     parser.add_argument("-m", "--model-name", default='distilgpt2', type=str)
#     args = parser.parse_args()
#
#     run(args.subreddit, args.model_month, args.model_name)
