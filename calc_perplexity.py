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




_DESCRIPTION = "This is a modified version of huggingface perplexity that outputs loss at each token."
_CITATION = ""
_KWARGS_DESCRIPTION = ""


# https://github.com/huggingface/evaluate/blob/main/measurements/perplexity/perplexity.py
class myPerplexity(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "data": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
            self, data, model_id, batch_size: int = 16, add_start_token: bool = False, device=None, max_length=None
    ):

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                    tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            data,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        sentence_ppls = []
        token_ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            # the following parts are different from the original code

            token_CEloss = loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch

            ppl_by_token = torch.exp(torch.div(token_CEloss.T, shift_attention_mask_batch.sum(1))).T

            batch_token_ppls = [
                list(
                    zip(
                        [x for x in encoded_batch.cpu().numpy()[i] if x != tokenizer.eos_token_id],
                        ppl_by_token.cpu().numpy()[i]
                    )
                )
                for i in range(encoded_batch.shape[0])
            ]

            perplexity_batch = torch.prod(ppl_by_token, 1)

            sentence_ppls += perplexity_batch.tolist()
            token_ppls += batch_token_ppls
        return {"sentence_ppls": sentence_ppls, "token_ppls": token_ppls}
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-sr", "--subreddit", required=True, type=str)
#     parser.add_argument("-month", "--model-month", required=True, type=str)
#     parser.add_argument("-m", "--model-name", default='distilgpt2', type=str)
#     args = parser.parse_args()
#
#     run(args.subreddit, args.model_month, args.model_name)
