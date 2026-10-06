"""Signals that tell whether retrieval found an answer, built from the saved retrieval runs.

For every question:
- labels:  answerable  (the knowledge base contains an answer)
           resolvable  (the correct article is among the top CONTEXT_DOCS of the primary retriever,
                        so an answer step reading those articles could solve it)
- features per retriever (dense, bm25, hybrid, rerankers): best score, gap to the second, how far the best
  score stands out from the rest of the list
- agreement: do keyword search, embeddings and rerankers point at the same article?
- question shape: length, lines, share of code-like characters
"""

import numpy as np
import pandas as pd

from src import config

SHORT = {"dense": "dense", "bm25": "bm25", "hybrid": "hybrid",
         "hybrid+rerank:bge-reranker-base": "rr_bge", "hybrid+rerank:ms-marco-MiniLM-L-6-v2": "rr_minilm"}


def short_name(method: str) -> str:
    return SHORT.get(method, method.replace("hybrid+rerank:", "rr_").replace("-", "_").replace(".", "_"))


def score_features(scores: list[float], prefix: str) -> dict:
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return {f"{prefix}_top1": np.nan, f"{prefix}_gap12": np.nan, f"{prefix}_z1": np.nan}
    rest = s[1:]
    spread = rest.std() if len(rest) > 1 else 0.0
    return {
        f"{prefix}_top1": s[0],
        f"{prefix}_gap12": s[0] - s[1] if len(s) > 1 else 0.0,
        f"{prefix}_z1": (s[0] - rest.mean()) / (spread + 1e-6) if len(rest) else 0.0,  # how much the best stands out
    }


def rank_in(doc: str | None, ranked: list[str], cap: int) -> int:
    if doc is None:
        return cap + 1
    try:
        return ranked.index(doc) + 1
    except ValueError:
        return cap + 1


def question_features(text: str) -> dict:
    words = text.split()
    code_chars = sum(text.count(c) for c in "{}[]()<>=/\\_:;$#|`")
    return {
        "q_log_words": np.log1p(len(words)),
        "q_lines": text.count("\n") + 1,
        "q_code_share": code_chars / max(len(text), 1),
    }


def build_features(runs: pd.DataFrame, queries: pd.DataFrame, primary: str = config.PRIMARY_RETRIEVER,
                   context_docs: int = config.CONTEXT_DOCS, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per question: labels plus features. `runs` is the retrieval output (one row per question x method).

    `extra` adds per-question signals from later steps, e.g. the LLM judgement (query_id, llm_p_yes, ...).
    """
    methods = list(dict.fromkeys(runs["method"]))
    if primary not in methods:
        raise ValueError(f"primary retriever {primary!r} not in runs: {methods}")
    by_method = {m: part.set_index("query_id") for m, part in runs.groupby("method", sort=False)}
    q = queries.set_index("query_id")
    ids = by_method[primary].index
    cap = max(len(d) for d in by_method[primary]["doc_ids"]) if len(ids) else 20

    rows = []
    for qid in ids:
        lists = {m: list(by_method[m].at[qid, "doc_ids"]) for m in methods if qid in by_method[m].index}
        scores = {m: list(by_method[m].at[qid, "scores"]) for m in lists}
        gold = set(q.at[qid, "gold_doc_ids"])
        primary_top = lists[primary]
        answerable = bool(q.at[qid, "answerable"])
        # An unanswerable question is never resolvable, even if its record lists related documents
        row = {"query_id": qid, "split": q.at[qid, "split"], "answerable": answerable,
               "resolvable": answerable and bool(gold & set(primary_top[:context_docs]))}
        for m in methods:
            if m in lists:
                row.update(score_features(scores[m], short_name(m)))
        top1 = primary_top[0] if primary_top else None
        for m in methods:
            if m != primary and m in lists:
                row[f"agree_{short_name(m)}_rank_of_{short_name(primary)}_top1"] = rank_in(top1, lists[m], cap)
        if "bm25" in lists:
            a, b = set(primary_top[:5]), set(lists["bm25"][:5])
            row["agree_top5_overlap_bm25"] = len(a & b) / 5
        firsts = [lst[0] for lst in lists.values() if lst]
        row["agree_methods_same_top1"] = sum(d == top1 for d in firsts) / max(len(firsts), 1)
        row.update(question_features(str(q.at[qid, "text"])))
        rows.append(row)
    feats = pd.DataFrame(rows)
    if extra is not None:
        cols = [c for c in extra.columns if c.startswith("llm_")]
        feats = feats.merge(extra[["query_id", *cols]], on="query_id", how="left")
        if "llm_p_yes" in cols:
            p = feats["llm_p_yes"].clip(1e-4, 1 - 1e-4)
            feats["llm_logit"] = np.log(p / (1 - p))  # log-odds: linear models use it better than the raw probability
    return feats


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ("query_id", "split", "answerable", "resolvable")]
