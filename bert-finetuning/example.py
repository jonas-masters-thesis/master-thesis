from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from myutils import make_logging


def main():
    make_logging('finetune-example')
    raw_datasets = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['text']), batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    training_args = TrainingArguments("example_trainer")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
    )
    trainer.train()


if __name__ == '__main__':
    main()
