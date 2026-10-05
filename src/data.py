"""Download, clean and split the support-ticket dataset.

Run:  python -m src.data
Writes data/processed/{train,val,test}.parquet and results/data_summary.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(source: str = config.DATASET_ID) -> pd.DataFrame:
    """Load tickets from a local CSV/Parquet file or from the Hugging Face Hub."""
    path = Path(source)
    if path.exists():
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    from datasets import load_dataset  # imported here so tests don't need network access

    return load_dataset(source, split="train").to_pandas()


def clean(df: pd.DataFrame, language: str | None = config.LANGUAGE) -> pd.DataFrame:
    """Keep one language, build a single text field and drop unusable rows."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if language and "language" in df.columns:
        df = df[df["language"].str.lower() == language]

    subject = df["subject"].fillna("").astype(str).str.strip() if "subject" in df else ""
    body = df["body"].fillna("").astype(str).str.strip()
    df["text"] = (subject + "\n\n" + body).str.strip()

    label_cols = [col for col in config.TASKS.values() if col in df.columns]
    for col in label_cols:
        df[col] = df[col].astype("string").str.strip()

    df = df[df["text"].str.len() > 0].dropna(subset=label_cols)
    # Exact duplicates would leak between train and test and inflate the scores.
    df = df.drop_duplicates(subset="text")
    return df[["text", *label_cols]].reset_index(drop=True)


def split(df: pd.DataFrame, stratify_col: str = "queue", seed: int = config.SEED):
    """Stratified train / validation / test split, so every queue appears in each part."""
    strat = df[stratify_col] if stratify_col in df else None
    train_val, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed, stratify=strat)

    strat = train_val[stratify_col] if stratify_col in df else None
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_share, random_state=seed, stratify=strat)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train, val, test, out_dir: Path | None = None) -> None:
    out_dir = out_dir or config.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)


def load_splits(data_dir: Path | None = None):
    data_dir = data_dir or config.DATA_DIR
    parts = []
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
        parts.append(pd.read_parquet(path))
    return tuple(parts)


PRIORITY_ORDER = ["very low", "low", "medium", "high", "critical", "urgent"]


def label_names(df: pd.DataFrame, task: str) -> list[str]:
    """Classes for a task. Priorities go from least to most urgent; everything else is alphabetical."""
    labels = sorted(df[config.TASKS[task]].unique().tolist())
    if task == "priority":
        def rank(label: str) -> int:
            key = label.lower().replace("_", " ")
            return PRIORITY_ORDER.index(key) if key in PRIORITY_ORDER else len(PRIORITY_ORDER)
        labels.sort(key=rank)  # stable sort keeps unknown labels alphabetical at the end
    return labels


def summarise(train, val, test) -> dict:
    full = pd.concat([train, val, test])
    summary = {
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "median_text_words": int(full["text"].str.split().str.len().median()),
        "label_counts": {},
    }
    for task, col in config.TASKS.items():
        if col in full:
            summary["label_counts"][task] = full[col].value_counts().to_dict()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=config.DATASET_ID, help="HF dataset id or local CSV/Parquet file")
    parser.add_argument("--language", default=config.LANGUAGE, help="language code to keep, or 'all'")
    args = parser.parse_args()

    raw = load_raw(args.source)
    df = clean(raw, None if args.language == "all" else args.language)
    train, val, test = split(df)
    save_splits(train, val, test)

    summary = summarise(train, val, test)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Loaded {len(raw):,} raw tickets -> {len(df):,} after cleaning")
    print(f"Split: {summary['rows']}")
    for task, counts in summary["label_counts"].items():
        print(f"\n{task}:")
        for label, n in counts.items():
            print(f"  {label:<35} {n:>6,}")


if __name__ == "__main__":
    main()
