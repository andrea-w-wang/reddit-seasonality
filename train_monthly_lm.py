import argparse
import evaluate
import pickle as pk
import torch
# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score
# Transformer library
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

data_dir = "./data/"
device_name = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset, Dataset


class SubredditMonthModel:
    def __init__(self, subreddit, month, model_name):
        self.subreddit = subreddit
        self.model_name = model_name
        self.month = month
        self.model_output_path = f"models/{self.model_name}_{self.subreddit}_{self.month}"
        self.lm_dataset = None

        self.model = None
        self.tokenizer = None

    def pipeline(self):
        print(f"Start: {self.model_output_path}...")
        print("- Load tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("- Processing data...")
        self.lm_dataset = self._prep_data_for_finetuning()
        print("- Finetuning...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device_name)
        self.train()

    def _prep_data_for_finetuning(self):
        usecols = ['year-month', 'timestamp', 'text', 'speaker']
        comments_df = pk.load(open(data_dir + f"{self.subreddit}-comments.pk", "rb"))
        comments_df = comments_df[usecols]
        monthly_comments_df = comments_df[comments_df['year-month'] == self.month]
        monthly_comments = Dataset.from_pandas(monthly_comments_df)

        monthly_comments = monthly_comments.train_test_split(test_size=0.2)

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        tokenized_monthly_comments = monthly_comments.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=monthly_comments["train"].column_names,
        )

        block_size = 128

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = tokenized_monthly_comments.map(group_texts, batched=True, num_proc=4)
        return lm_dataset

    def train(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=self.model_output_path,
            num_train_epochs=10,
            evaluation_strategy="steps",
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir='./logs',  # directory for storing logs
            logging_steps=50,
            eval_steps=50,
            load_best_model_at_end=True,
            save_steps=100
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["test"],
            data_collator=data_collator,
        )
        trainer.train()
        self.model.save_pretrained(self.model_output_path + "./best")
        self.tokenizer.save_pretrained(self.model_output_path + "./best")


if __name__ == '__main__':
    from dateutil.parser import parse
    from dateutil.relativedelta import relativedelta

    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str)
    parser.add_argument("-start", "--start-month", default='2016-01', type=str)
    parser.add_argument("-end", "--end-month", default='2018-01', type=str)
    parser.add_argument("-m", "--model", required=True, type=str)
    args = parser.parse_args()

    this_month = args.start_month
    while parse(this_month) <= parse(args.end_month):
        m = SubredditMonthModel(args.subreddit, this_month, args.model)
        m.pipeline()
        this_month = (parse(this_month) + relativedelta(months=1)).strftime("%Y-%m")
