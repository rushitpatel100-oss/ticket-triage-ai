"""Fine-tune a Hugging Face transformer (DistilBERT by default) for each ticket task.

Run on a GPU (e.g. Google Colab, see notebooks/train_on_colab.ipynb):
    python -m src.train                       # all tasks
    python -m src.train --task queue --epochs 2
    python -m src.train --max-train-samples 3000   # quick trial run

Optional: --push-to-hub your-hf-username/ticket-triage  uploads each model to the Hub
(log in first with `huggingface-cli login`; never put your token in the code).
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src import config
from src.data import label_names, load_splits
from src.evaluate import compute_metrics, log_result, plot_confusion_matrix, save_errors


class WeightedTrainer(Trainer):
    """Trainer whose loss gives rare classes more weight (the same idea as class_weight="balanced" in scikit-learn)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def balanced_class_weights(labels_idx, n_classes: int) -> torch.Tensor:
    """weight = n_samples / (n_classes * count), so every class counts equally overall."""
    counts = np.bincount(labels_idx, minlength=n_classes).astype(float)
    return torch.tensor(len(labels_idx) / (n_classes * np.maximum(counts, 1)), dtype=torch.float)


def display_name(model_name: str) -> str:
    short = model_name.rstrip("/").split("/")[-1]
    return "DistilBERT (fine-tuned)" if short.startswith("distilbert") else f"{short} (fine-tuned)"


def train_task(task: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, args) -> dict:
    col = config.TASKS[task]
    labels = label_names(pd.concat([train, val, test]), task)  # every class, even rare ones
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"\n=== {task}: {len(labels)} classes, {len(train):,} training tickets ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label2id,
    )

    def to_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(pd.DataFrame({"text": df["text"].tolist(), "label": df[col].map(label2id).tolist()}))
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    def metrics_fn(eval_pred):
        logits, y = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(y, preds), "macro_f1": f1_score(y, preds, average="macro")}

    training_args = TrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints" / task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.06,  # a float means "6% of all steps"
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,  # keep the epoch with the best validation macro-F1
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=config.SEED,
    )
    class_weights = None
    if args.class_weights:
        class_weights = balanced_class_weights(train[col].map(label2id).to_numpy(), len(labels))
        print("Class weights:", {lab: round(float(w), 2) for lab, w in zip(labels, class_weights)})
    trainer = WeightedTrainer(
        model=model,
        class_weights=class_weights,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn,
    )

    start = time.time()
    trainer.train()
    train_seconds = round(time.time() - start)

    # Final, untouched test set
    logits = trainer.predict(test_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = [labels[i] for i in probs.argmax(axis=1)]

    name = display_name(args.model_name)
    metrics = compute_metrics(test[col], preds, labels)
    metrics["params"] = {"base_model": args.model_name, "epochs": args.epochs, "batch_size": args.batch_size,
                         "lr": args.lr, "max_length": args.max_length, "train_rows": len(train),
                         "class_weights": bool(args.class_weights)}
    metrics["train_seconds"] = train_seconds
    metrics["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log_result(task, name, metrics)
    print(f"{name}  acc={metrics['accuracy']:.3f}  macro-F1={metrics['macro_f1']:.3f}  ({train_seconds}s)")

    plot_confusion_matrix(test[col], preds, labels, f"{name}: {task}", config.RESULTS_DIR / f"confusion_{task}_transformer.png")
    save_errors(test["text"], test[col], preds, probs.max(axis=1), config.RESULTS_DIR / f"errors_{task}_transformer.csv")

    out_dir = config.MODELS_DIR / task
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved model to {out_dir}")

    if args.push_to_hub:
        repo_id = f"{args.push_to_hub}-{task}"
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Uploaded to https://huggingface.co/{repo_id}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(config.TASKS), help="train one task only (default: all)")
    parser.add_argument("--model-name", default=config.BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false",
                        help="train with the plain loss (rare classes are then easy to ignore)")
    parser.add_argument("--max-train-samples", type=int, help="use a random subset of the training data")
    parser.add_argument("--push-to-hub", metavar="USER/PREFIX", help="upload models as USER/PREFIX-<task>")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: no GPU found. Training will be very slow; use Google Colab (see README).")

    train, val, test = load_splits()
    if args.max_train_samples and args.max_train_samples < len(train):
        train = train.sample(args.max_train_samples, random_state=config.SEED).reset_index(drop=True)

    tasks = [args.task] if args.task else [t for t, col in config.TASKS.items() if col in train]
    for task in tasks:
        train_task(task, train, val, test, args)


if __name__ == "__main__":
    main()
