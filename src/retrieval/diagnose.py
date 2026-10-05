"""Why do the rerankers lose to plain embedding search? Diagnostics on the saved retrieval runs.

1. Where the correct document moves when the reranker re-orders the hybrid candidates.
2. Reciprocal rank by question length (quartiles), for each method.
3. Length of the top document each method returns, against the correct document's length.
4. A controlled test (--truncation-test, needs a GPU): rerank the same 20 candidates twice, once with the
   full question and once with only its first SHORT_WORDS words. If long questions crowd the passage out
   of the cross-encoder's 512-token window, the short version should do better.
5. A few examples where the reranker pushed the correct document down.

Run after `python -m src.retrieval`:
    python -m src.retrieval.diagnose --truncation-test
Writes results/retrieval_diagnostics.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.retrieval.metrics import per_query
from src.sources.common import load_tables

SHORT_WORDS = 64
DEV = ("train", "val")


def rank_of_gold(doc_ids, gold) -> float:
    for i, d in enumerate(doc_ids):
        if d in gold:
            return i + 1
    return np.inf


def load_runs(name: str):
    docs, queries = load_tables(name)
    runs = pd.read_parquet(config.DATA_DIR / "retrieval" / f"{name}_runs.parquet")
    runs = runs[runs["answerable"] & runs["split"].isin(DEV)]
    q = queries.set_index("query_id")
    return docs.set_index("doc_id"), q, runs


def describe(name: str) -> dict:
    docs, q, runs = load_runs(name)
    wide = runs.pivot(index="query_id", columns="method", values="doc_ids")
    gold = q.loc[wide.index, "gold_doc_ids"].map(set)
    q_words = q.loc[wide.index, "text"].str.split().str.len()
    doc_words = docs["text"].str.split().str.len()
    out = {"dev_questions": int(len(wide)), "question_words_median": int(q_words.median())}

    ranks = {m: np.array([rank_of_gold(list(d), g) for d, g in zip(wide[m], gold)]) for m in wide.columns}
    rr = {m: per_query([list(d) for d in wide[m]], [list(g) for g in gold])["rr"] for m in wide.columns}

    # 1. Movement of the correct document, hybrid -> each reranker (only where hybrid found it in its top 20)
    out["movement_vs_hybrid"] = {}
    for m in [c for c in wide.columns if c.startswith("hybrid+rerank")]:
        found = np.isfinite(ranks["hybrid"])
        h, r = ranks["hybrid"][found], ranks[m][found]
        out["movement_vs_hybrid"][m] = {
            "found_by_hybrid": int(found.sum()),
            "moved_up": int((r < h).sum()), "same": int((r == h).sum()), "moved_down": int((r > h).sum()),
            "was_first_now_below_5": int(((h == 1) & (r > 5)).sum()), "hybrid_first": int((h == 1).sum()),
        }

    # 2. Reciprocal rank by question length quartile
    bins = pd.qcut(q_words, 4, labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"], duplicates="drop")
    edges = q_words.groupby(bins, observed=True).agg(["min", "max"])
    out["mrr_by_question_length"] = {
        str(b): {"words": f"{int(edges.loc[b, 'min'])}-{int(edges.loc[b, 'max'])}", "n": int((bins == b).sum()),
                 **{m: round(float(rr[m][(bins == b).to_numpy()].mean()), 4) for m in wide.columns}}
        for b in edges.index}

    # 3. Length of the top document per method
    top1 = {m: wide[m].map(lambda d: d[0] if len(d) else None) for m in wide.columns}
    out["top_document_words_median"] = {m: int(doc_words.reindex(t.dropna()).median()) for m, t in top1.items()}
    gold_first = gold.map(lambda g: sorted(g)[0])
    out["top_document_words_median"]["correct document"] = int(doc_words.reindex(gold_first).median())

    # 5. Examples where a reranker demoted the correct document from 1st to below 5th
    out["examples"] = []
    for m in [c for c in wide.columns if c.startswith("hybrid+rerank")][:1]:
        bad = np.flatnonzero((ranks["hybrid"] == 1) & (ranks[m] > 5))[:3]
        for i in bad:
            qid = wide.index[i]
            out["examples"].append({
                "method": m, "question": q.loc[qid, "text"][:400],
                "correct_document": docs.loc[next(iter(gold.iloc[i])), "text"][:300],
                "reranker_top_document": docs.loc[wide[m].iloc[i][0], "text"][:300],
                "correct_rank_after_rerank": float(ranks[m][i]),
            })
    return out


def truncation_test(name: str, reranker_names, max_questions: int = 1000, seed: int = config.SEED) -> dict:
    """Same candidates, same documents: full question vs its first SHORT_WORDS words."""
    from src.retrieval.retrievers import CrossEncoderReranker

    docs, q, runs = load_runs(name)
    hybrid = runs[runs["method"] == "hybrid"].set_index("query_id")["doc_ids"]
    hybrid = hybrid[hybrid.map(len) > 0]
    if len(hybrid) > max_questions:
        hybrid = hybrid.sample(max_questions, random_state=seed)
    gold = [list(q.loc[qid, "gold_doc_ids"]) for qid in hybrid.index]
    full = q.loc[hybrid.index, "text"].tolist()
    short = [" ".join(t.split()[:SHORT_WORDS]) for t in full]

    out = {"questions": len(hybrid), "short_words": SHORT_WORDS,
           "hybrid_mrr@10": round(float(per_query([list(d) for d in hybrid], gold)["rr"].mean()), 4)}
    for model_name in reranker_names:
        reranker = CrossEncoderReranker(model_name, fp16=True)
        for label, texts in (("full_question", full), ("first_words", short)):
            pairs, owners = [], []
            for i, (text, cands) in enumerate(zip(texts, hybrid)):
                for d in cands:
                    pairs.append((text, docs.loc[d, "text"]))
                    owners.append(i)
            scores = reranker.score(pairs)
            ranked, pos = [], 0
            for cands in hybrid:
                s = scores[pos:pos + len(cands)]
                pos += len(cands)
                ranked.append([cands[j] for j in np.argsort(-s, kind="stable")])
            out[f"{model_name.split('/')[-1]}:{label}_mrr@10"] = round(float(per_query(ranked, gold)["rr"].mean()), 4)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["techqa", "stackexchange"])
    parser.add_argument("--truncation-test", action="store_true")
    parser.add_argument("--rerankers", nargs="+",
                        default=["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"])
    args = parser.parse_args(argv)
    results = {}
    for name in args.datasets:
        if not (config.DATA_DIR / "retrieval" / f"{name}_runs.parquet").exists():
            continue
        results[name] = describe(name)
        if args.truncation_test:
            results[name]["truncation_test"] = truncation_test(name, args.rerankers)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "retrieval_diagnostics.json").write_text(json.dumps(results, indent=2))
    print("DIAG_START")
    print(json.dumps(results))
    print("DIAG_END")


if __name__ == "__main__":
    main()
