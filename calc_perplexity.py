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

    return perplexity.compute(
        model_id=checkpoint_path,
        add_start_token=False,  # default
        data=sentences,
        max_length=1024
    )


def run(subreddit, model_month, model_name="distilgpt2"):
    usecols = ['year-month', 'text']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]

    perplexity = metrics.myPerplexity()
    checkpoint_name = f"{model_name}_{subreddit}_{model_month}"
    checkpoint = f"./models/{checkpoint_name}/best"
    print(checkpoint_name)

    last_month = "2018-10"
    predict_month = "2014-01"
    while predict_month <= last_month:

        print(f"***Predict {predict_month}***")
        input_texts = [t for t in comments_df[comments_df['year-month'] == predict_month]['text'] if t]
        selected_random_utts = list(np.random.choice(input_texts, size=300, replace=False))
        del input_texts
        ppl_month = perplexity._compute(
            model_id=checkpoint,
            add_start_token=False,
            data=selected_random_utts,
            max_length=1024
        )
        ppl_month['utterances'] = selected_random_utts
        pk.dump(ppl_month,
                open(data_dir + f"output/{checkpoint_name}_predict{predict_month}_scores.pk", "wb"))

        predict_month = (parse(predict_month) + relativedelta(months=1)).strftime("%Y-%m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--subreddit", required=True, type=str)
    parser.add_argument("-month", "--model-month", required=True, type=str)
    parser.add_argument("-model", "--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()
    run(args.subreddit, args.model_month, args.model_name)
