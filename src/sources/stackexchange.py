"""Stack Exchange (Super User, Ask Ubuntu): real tech-support questions with accepted answers.

Source: HuggingFaceH4/stack-exchange-preferences (CC BY-SA 4.0, attribution required; every document keeps
a link to the original answer).

Design choices:
- A knowledge document is the accepted answer only. The question text is the query, so putting it in the
  document as well would make retrieval artificially easy (the query would match itself).
- To get realistic "the knowledge base cannot answer this" cases, a share of questions (KB_HOLDOUT_SHARE)
  has its accepted answer removed from the knowledge base and is labelled answerable=False. A similar
  answer may still exist elsewhere in the base, so these labels are slightly noisy; the data card says so.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.sources.common import DOC_COLUMNS, QUERY_COLUMNS, assign_splits, first_line, raw_dir, strip_html

NAME = "stackexchange"
MIN_CHARS = 20


def site_files(site: str, max_files: int) -> list[str]:
    from huggingface_hub import list_repo_files  # imported here so tests stay offline

    files = sorted(f for f in list_repo_files(config.STACKEXCHANGE_REPO, repo_type="dataset")
                   if f.startswith(f"data/{site}/") and f.endswith(".parquet"))
    if not files:
        raise ValueError(f"no parquet files for {site} in {config.STACKEXCHANGE_REPO}")
    return files[:max_files]


def download(sites=config.STACKEXCHANGE_SITES, max_files: int = config.STACKEXCHANGE_MAX_FILES,
             out_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir = out_dir or raw_dir(NAME)
    for site in sites:
        for filename in site_files(site, max_files):
            hf_hub_download(config.STACKEXCHANGE_REPO, filename, repo_type="dataset", local_dir=out_dir)
    return out_dir


def _is_selected(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def accepted_pairs(raw: pd.DataFrame, site: str) -> pd.DataFrame:
    """One row per question that has an accepted answer: question text, answer text and ids."""
    short = site.split(".")[0]
    rows = []
    for qid, question, answers in zip(raw["qid"], raw["question"], raw["answers"]):
        accepted = [a for a in (answers if answers is not None else []) if _is_selected(a.get("selected"))]
        if not accepted:
            continue
        answer = accepted[0]
        rows.append({
            "query_id": f"se-{short}:q{qid}",
            "doc_id": f"se-{short}:a{answer['answer_id']}",
            "question": strip_html(question),
            "answer": strip_html(answer.get("text", "")),
            "url": f"https://{site}/a/{answer['answer_id']}",
        })
    pairs = pd.DataFrame(rows, columns=["query_id", "doc_id", "question", "answer", "url"])
    pairs = pairs[(pairs["question"].str.len() >= MIN_CHARS) & (pairs["answer"].str.len() >= MIN_CHARS)]
    return pairs.drop_duplicates("question").drop_duplicates("doc_id").reset_index(drop=True)


def load(sites=config.STACKEXCHANGE_SITES, max_questions: int = config.STACKEXCHANGE_MAX_QUESTIONS,
         holdout_share: float = config.KB_HOLDOUT_SHARE, in_dir: Path | None = None,
         seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_dir = in_dir or raw_dir(NAME)
    per_site = []
    for site in sites:
        files = sorted((in_dir / "data" / site).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files in {in_dir / 'data' / site}. Run the download first.")
        raw = pd.concat([pd.read_parquet(f, columns=["qid", "question", "answers"]) for f in files], ignore_index=True)
        pairs = accepted_pairs(raw, site)
        if len(pairs) > max_questions:
            pairs = pairs.sample(max_questions, random_state=seed)
        pairs["source"] = f"Stack Exchange ({site})"
        per_site.append(pairs)
    pairs = pd.concat(per_site, ignore_index=True)

    # Knowledge-base gaps: remove the answer of a random share of questions.
    rng = np.random.default_rng(seed)
    held_out = rng.random(len(pairs)) < holdout_share

    queries = pd.DataFrame({
        "query_id": pairs["query_id"],
        "source": pairs["source"],
        "text": pairs["question"],
        "answerable": ~held_out,
        "gold_doc_ids": [[d] if not h else [] for d, h in zip(pairs["doc_id"], held_out)],
        "gold_answer": pairs["answer"],
    })
    queries["split"] = assign_splits(queries)

    kept = pairs[~held_out]
    docs = pd.DataFrame({
        "doc_id": kept["doc_id"],
        "source": kept["source"],
        "title": kept["answer"].map(first_line),
        "text": kept["answer"],
        "url": kept["url"],
    }).reset_index(drop=True)
    return docs[DOC_COLUMNS], queries[QUERY_COLUMNS]
