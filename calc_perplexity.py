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

data_dir = "./data/"
device_name = "cuda" if torch.cuda.is_available() else "cpu"


# https://huggingface.co/spaces/evaluate-metric/perplexity

def run(subreddit, model_month, model_name="distilgpt2"):
    usecols = ['year-month', 'timestamp', 'text', 'speaker']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]

    perplexity = evaluate.load("perplexity", module_type="metric")
    checkpoint_name = f"{model_name}_{subreddit}_{model_month}"
    checkpoint = f"./models/{checkpoint_name}/best"
    print(checkpoint_name)
    max_length = 512

    ppl_results = dict()

    predict_month = model_month
    for i in range(1, 25):
        predict_month = (parse(predict_month) + relativedelta(months=1)).strftime("%Y-%m")
        print(f"***Predict {predict_month}***")
        input_texts = [t[:max_length] for t in
                       comments_df[comments_df['year-month'] == predict_month]['text']
                       if t != ""]
        if len(input_texts) == 0:
            print(f"{predict_month}: no data")
            continue
        ppl_results[predict_month] = perplexity.compute(
            model_id=checkpoint,
            add_start_token=True,  # default
            predictions=input_texts
        )
    pk.dump(ppl_results,
            open(data_dir + f"output/{checkpoint_name}_scores.pk", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str)
    parser.add_argument("-month", "--model-month", required=True, type=str)
    parser.add_argument("-m", "--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()

    run(args.subreddit, args.model_month, args.model_name)
