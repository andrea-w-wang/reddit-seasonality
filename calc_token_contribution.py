import argparse
import evaluate
import pickle as pk
import torch
from collections import defaultdict
from datasets import load_dataset, Dataset
# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score
# Transformer library
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

data_dir = "./data/"
device_name = "cuda" if torch.cuda.is_available() else "cpu"


def calc_token_nlls(model, encodings, max_length=512):
    stride = 1
    seq_len = encodings.size(1)

    nlls = defaultdict(list)
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        print("begin_loc: ", begin_loc)
        end_loc = min(begin_loc + max_length, seq_len)
        print("end_loc: ", end_loc)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings[:, begin_loc:end_loc].to(device_name)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_token = target_ids[:, -stride]
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            token_nll = outputs.loss * trg_len

            target_word = tokenizer.decode(target_token)
            nlls[target_word].append(token_nll.cpu().item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls


def run(subreddit, model_month, predict_month):
    usecols = ['year-month', 'timestamp', 'text', 'speaker']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]

    checkpoint_path = f"./models/distilgpt2_{subreddit}_{model_month}/best"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    input_texts = [t for t in comments_df[comments_df['year-month'] == predict_month]['text'] if t]
    encodings = tokenizer.encode("\n\n".join(input_texts), return_tensors="pt")
    nlls = calc_token_nlls(model, encodings)
    pk.dump(nlls, open(f"data/output/{subreddit}-nll-model={model_month}-predict={predict_month}.pk", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--subreddit", required=True, type=str)
    parser.add_argument("--model-month", required=True, type=str)
    parser.add_argument("--predict-month", required=True, type=str)
    parser.add_argument("-m", "--model-name", default='distilgpt2', type=str)
    args = parser.parse_args()

    run(args.subreddit, args.model_month, args.predict_month)
