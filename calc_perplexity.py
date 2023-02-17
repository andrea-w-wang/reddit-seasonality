import argparse
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


def calculate_my_ppl(checkpoint_path, sentences):
    perplexity = metrics.myPerplexity()
    return perplexity.compute(
        model_id=checkpoint_path,
        add_start_token=False,  # default
        data=sentences,
        max_length=1024
    )


def run(subreddit, predict_month, model_month, model_name="distilgpt2"):
    usecols = ['year-month', 'text']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]

    input_texts = [t for t in comments_df[comments_df['year-month'] == predict_month]['text'] if t]
    selected_random_utts = list(np.random.choice(input_texts, size=300, replace=False))

    ppl_results = {"predict_month": predict_month,
                   "utterances": selected_random_utts,
                   "model_month": model_month
                   }

    checkpoint_path = f"./models/{model_name}_{subreddit}_{model_month}/best"
    my_ppl = calculate_my_ppl(checkpoint_path, selected_random_utts)
    ppl_results.update(my_ppl)

    return ppl_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--subreddit", required=True, type=str)
    parser.add_argument("-pm", "--predict-month", required=True, type=str)
    parser.add_argument("-smm", "--start-model-month", required=True, type=str)
    parser.add_argument("-emm", "--end-model-month", required=True, type=str)
    parser.add_argument("--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()

    model_month = args.start_model_month

    while parse(model_month) <= parse(args.end_model_month):
        print(model_month)
        output = run(args.subreddit, args.predict_month, model_month)
        pk.dump(output, open(
            f"./data/output/{args.subreddit}-model{model_month}-predict{args.predict_month}.pk", "wb"
        ))

        model_month = (parse(model_month) + relativedelta(months=1)).strftime("%Y-%m")
