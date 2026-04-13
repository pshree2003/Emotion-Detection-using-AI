from __future__ import annotations

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_CHECKPOINT = "distilroberta-base"
OUTPUT_DIR = "./models/text-emotion-model"
MAX_SAMPLES = 20000

# A practical subset of emotion classes from GoEmotions.
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]


def main() -> None:
    dataset = load_dataset("go_emotions", "raw")

    # Flatten multi-label rows to a single representative label for demo training.
    def pick_first_label(example):
        labels = example["labels"]
        example["label"] = labels[0] if labels else 27
        return example

    mapped = dataset["train"].map(pick_first_label)
    mapped = mapped.filter(lambda x: x["label"] in [17, 25, 2, 14, 26, 27])
    mapped = mapped.shuffle(seed=42).select(range(min(MAX_SAMPLES, len(mapped))))

    id_mapping = {17: 0, 25: 1, 2: 2, 14: 3, 26: 4, 27: 5}
    mapped = mapped.map(lambda x: {"label": id_mapping[x["label"]]})

    split = mapped.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_train = split["train"].map(tokenize, batched=True)
    tokenized_eval = split["test"].map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(EMOTION_LABELS),
        id2label={i: l for i, l in enumerate(EMOTION_LABELS)},
        label2id={l: i for i, l in enumerate(EMOTION_LABELS)},
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
