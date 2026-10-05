"""Shared helpers for the real-world data sources.

Every knowledge source is turned into the same two tables, so retrieval and the resolve-or-escalate model
can treat them alike:

docs     one row per knowledge document
         doc_id, source, title, text, url
queries  one row per user question / ticket
         query_id, source, text, answerable, gold_doc_ids, gold_answer, split
"""

import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

DOC_COLUMNS = ["doc_id", "source", "title", "text", "url"]
QUERY_COLUMNS = ["query_id", "source", "text", "answerable", "gold_doc_ids", "gold_answer", "split"]

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|div|br|li|ul|ol|pre|h[1-6]|blockquote|tr)\b[^>]*>", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def raw_dir(name: str) -> Path:
    return config.RAW_DIR / name


def processed_dir(name: str) -> Path:
    return config.DATA_DIR / name


def strip_html(text: str) -> str:
    """Turn HTML into plain text, keeping line breaks between blocks and the text inside code tags."""
    if not isinstance(text, str):
        return ""
    text = _BLOCK_TAG.sub("\n", text)
    text = _TAG.sub("", text)
    return normalise_whitespace(html.unescape(text))


def normalise_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def first_line(text: str, max_chars: int = 200) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()[:max_chars]
    return ""


def assign_splits(queries: pd.DataFrame, seed: int = config.SEED, val_size: float = 0.1,
                  test_size: float = 0.1) -> pd.Series:
    """Stratified train/val/test labels on `answerable`, so both outcomes appear in every part.

    With test_size=0 only train and val are assigned.
    """
    idx = queries.index.to_numpy()
    stratify = queries["answerable"].nunique() > 1
    if test_size > 0:
        rest, test = train_test_split(idx, test_size=test_size, random_state=seed,
                                      stratify=queries["answerable"] if stratify else None)
    else:
        rest, test = idx, []
    strat_rest = queries.loc[rest, "answerable"] if stratify else None
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), random_state=seed, stratify=strat_rest)
    labels = pd.Series("train", index=queries.index)
    labels.loc[val] = "val"
    labels.loc[test] = "test"
    return labels


def check_tables(docs: pd.DataFrame, queries: pd.DataFrame) -> None:
    """Fail loudly if a loader produced tables that later stages can't trust."""
    missing = set(DOC_COLUMNS) - set(docs.columns) | set(QUERY_COLUMNS) - set(queries.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if docs["doc_id"].duplicated().any():
        raise ValueError("duplicate doc_id values")
    if queries["query_id"].duplicated().any():
        raise ValueError("duplicate query_id values")
    known = set(docs["doc_id"])
    answerable = queries[queries["answerable"]]
    dangling = [g for gold in answerable["gold_doc_ids"] for g in gold if g not in known]
    if dangling:
        raise ValueError(f"{len(dangling)} gold documents of answerable queries are not in the knowledge base")


def save_tables(name: str, docs: pd.DataFrame, queries: pd.DataFrame) -> Path:
    check_tables(docs, queries)
    out = processed_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    docs[DOC_COLUMNS].to_parquet(out / "docs.parquet", index=False)
    queries[QUERY_COLUMNS].to_parquet(out / "queries.parquet", index=False)
    return out


def load_tables(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = processed_dir(name)
    if not (out / "docs.parquet").exists():
        raise FileNotFoundError(f"{out} not found. Run `python -m src.sources` first.")
    docs = pd.read_parquet(out / "docs.parquet")
    queries = pd.read_parquet(out / "queries.parquet")
    queries["gold_doc_ids"] = queries["gold_doc_ids"].map(list)
    return docs, queries


def summarise_tables(docs: pd.DataFrame, queries: pd.DataFrame) -> dict:
    words = docs["text"].str.split().str.len()
    q_words = queries["text"].str.split().str.len()
    return {
        "docs": int(len(docs)),
        "doc_median_words": int(words.median()) if len(docs) else 0,
        "queries": int(len(queries)),
        "query_median_words": int(q_words.median()) if len(queries) else 0,
        "answerable_share": round(float(queries["answerable"].mean()), 4) if len(queries) else None,
        "split_counts": {k: int(v) for k, v in queries["split"].value_counts().items()},
        "answerable_by_split": {k: round(float(v), 4) for k, v in queries.groupby("split")["answerable"].mean().items()},
    }
