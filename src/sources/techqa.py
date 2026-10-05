"""TechQA: real questions from IBM's developer support forums, answered from IBM Technotes.

We use the nvidia/TechQA-RAG-Eval release (Apache-2.0): `train.json` holds the questions, the reference
answer, an `is_impossible` flag (no answer exists in the documents) and the supporting documents;
`corpus.zip` holds the Technotes that make up the knowledge base.

This is the realistic test for "can the AI resolve this ticket by itself?": some questions genuinely have
no answer in the knowledge base, and the system has to notice that and escalate.
"""

import json
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from src import config
from src.sources.common import (
    DOC_COLUMNS,
    QUERY_COLUMNS,
    assign_splits,
    first_line,
    normalise_whitespace,
    raw_dir,
)

NAME = "techqa"
SOURCE = "TechQA (IBM)"
FILES = ("train.json", "corpus.zip")


def download(out_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download  # imported here so tests stay offline

    out_dir = out_dir or raw_dir(NAME)
    for filename in FILES:
        hf_hub_download(config.TECHQA_REPO, filename, repo_type="dataset", local_dir=out_dir)
    return out_dir


def doc_id_from_filename(filename: str) -> str:
    return f"{NAME}:{PurePosixPath(str(filename)).stem}"


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # JSON lines
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        for key in ("data", "rows", "train"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"unexpected JSON layout in {path}: keys {list(data)[:10]}")
    return data


def _read_corpus(path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and "__MACOSX" not in m]
        for member in members:
            if member.lower().endswith(".txt"):
                text = zf.read(member).decode("utf-8", errors="replace")
                rows.append({"doc_id": doc_id_from_filename(member), "text": text})
        if not rows:
            raise ValueError(f"no .txt documents in {path}; first members: {members[:5]}")
    return pd.DataFrame(rows)


def load(in_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_dir = in_dir or raw_dir(NAME)
    records = _read_records(in_dir / "train.json")

    queries, context_docs = [], []
    for rec in records:
        contexts = rec.get("contexts") or []
        gold = []
        for ctx in contexts:
            doc_id = doc_id_from_filename(ctx["filename"])
            gold.append(doc_id)
            context_docs.append({"doc_id": doc_id, "text": ctx.get("text", "")})
        queries.append({
            "query_id": f"{NAME}:{rec['id']}",
            "source": SOURCE,
            "text": normalise_whitespace(rec["question"]),
            "answerable": not bool(rec.get("is_impossible", False)),
            "gold_doc_ids": list(dict.fromkeys(gold)),
            "gold_answer": normalise_whitespace(rec.get("answer") or ""),
        })
    queries = pd.DataFrame(queries)

    # Knowledge base = every Technote in the corpus, plus any supporting document the corpus lacks.
    docs = pd.concat([_read_corpus(in_dir / "corpus.zip"), pd.DataFrame(context_docs)], ignore_index=True)
    docs = docs.drop_duplicates("doc_id", keep="first").reset_index(drop=True)
    docs["text"] = docs["text"].map(normalise_whitespace)
    docs = docs[docs["text"].str.len() > 0].reset_index(drop=True)
    docs["source"] = SOURCE
    docs["title"] = docs["text"].map(first_line)
    docs["url"] = ""

    queries["split"] = official_or_stratified_split(queries)
    return docs[DOC_COLUMNS], queries[QUERY_COLUMNS]


def official_or_stratified_split(queries: pd.DataFrame) -> pd.Series:
    """Use TechQA's own TRAIN/DEV ids when present (DEV becomes our test set), otherwise a stratified split."""
    raw_ids = queries["query_id"].str.split(":", n=1).str[1].str.upper()
    is_dev = raw_ids.str.startswith("DEV")
    is_train = raw_ids.str.startswith("TRAIN")
    if not (is_dev.any() and is_train.any() and (is_dev | is_train).all()):
        return assign_splits(queries)

    labels = pd.Series("test", index=queries.index)
    train_part = queries[is_train]
    labels.loc[train_part.index] = assign_splits(train_part, val_size=0.15, test_size=0)
    return labels
