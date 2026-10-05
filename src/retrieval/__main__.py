"""Compare retrieval methods on the real knowledge bases.

Run on a GPU (e.g. Google Colab, see notebooks/retrieval_colab.ipynb) after `python -m src.sources`:
    python -m src.retrieval                          # TechQA and Stack Exchange
    python -m src.retrieval --datasets techqa
    python -m src.retrieval --max-queries-per-split 2000

Methods:
    bm25            keyword search
    dense           embedding search (best passage per document)
    hybrid          reciprocal rank fusion of bm25 + dense
    hybrid+rerank   hybrid candidates re-scored by a cross-encoder

Writes results/retrieval_metrics.json, results/retrieval_comparison.png and, for the resolve-or-escalate
model, data/processed/retrieval/<dataset>_runs.parquet (top results and scores for every query).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.retrieval.core import Ranking, chunk_documents, doc_starts, reciprocal_rank_fusion
from src.retrieval.metrics import summarise
from src.retrieval.retrievers import BM25, DenseRetriever, rerank
from src.sources.common import load_tables

DATASETS = ("techqa", "stackexchange")
METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")
LABELS = {"bm25": "BM25 (keywords)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (RRF)",
          "hybrid+rerank": "Hybrid + reranker"}


def select_queries(queries: pd.DataFrame, max_per_split: int | None, seed: int = config.SEED) -> pd.DataFrame:
    if not max_per_split:
        return queries.reset_index(drop=True)
    parts = [g.sample(min(len(g), max_per_split), random_state=seed) for _, g in queries.groupby("split")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def run_dataset(name: str, encoder, reranker, max_per_split: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Index one knowledge base, run every method on its queries, return metrics and per-query runs."""
    docs, queries = load_tables(name)
    docs = docs.reset_index(drop=True)
    queries = select_queries(queries, max_per_split)
    texts = queries["text"].tolist()
    passages = chunk_documents(docs)
    starts = doc_starts(passages["doc_pos"].to_numpy(), len(docs))
    passage_texts = passages["text"].tolist()
    print(f"\n=== {name}: {len(docs):,} documents, {len(passages):,} passages, {len(queries):,} queries ===")

    rankings: dict[str, Ranking] = {}
    timing: dict[str, dict] = {}

    t = time.time()
    bm25 = BM25().fit(docs["text"].tolist())
    index_s = time.time() - t
    t = time.time()
    rankings["bm25"] = bm25.search(texts, config.CANDIDATES)
    timing["bm25"] = {"index_seconds": round(index_s, 1), "query_seconds": time.time() - t}

    t = time.time()
    dense = DenseRetriever(encoder, passage_texts, passages["doc_pos"].to_numpy(), starts)
    index_s = time.time() - t
    t = time.time()
    query_emb = encoder.encode_queries(texts)
    rankings["dense"] = dense.search(query_emb, config.CANDIDATES)
    timing["dense"] = {"index_seconds": round(index_s, 1), "query_seconds": time.time() - t}

    t = time.time()
    rankings["hybrid"] = reciprocal_rank_fusion([rankings["bm25"], rankings["dense"]])
    fusion_s = time.time() - t
    timing["hybrid"] = {"index_seconds": timing["bm25"]["index_seconds"] + timing["dense"]["index_seconds"],
                        "query_seconds": timing["bm25"]["query_seconds"] + timing["dense"]["query_seconds"] + fusion_s}

    if reranker is not None:
        t = time.time()
        rankings["hybrid+rerank"] = rerank(reranker, texts, query_emb, rankings["hybrid"], dense, passage_texts,
                                           config.RERANK_DEPTH, config.RERANK_CHUNKS_PER_DOC)
        timing["hybrid+rerank"] = {"index_seconds": timing["hybrid"]["index_seconds"],
                                   "query_seconds": timing["hybrid"]["query_seconds"] + time.time() - t}

    doc_ids = docs["doc_id"].to_numpy()
    answerable = queries["answerable"].to_numpy()
    gold = queries["gold_doc_ids"].tolist()
    metrics, runs = {}, []
    for method, ranking in rankings.items():
        top = ranking.top(config.RUN_DEPTH)
        ranked = [[doc_ids[i] for i in row if i >= 0] for row in top.idx]
        scores = [[float(s) for s, i in zip(srow, irow) if i >= 0] for srow, irow in zip(top.scores, top.idx)]
        runs.append(pd.DataFrame({"query_id": queries["query_id"], "split": queries["split"],
                                  "answerable": answerable, "method": method, "doc_ids": ranked, "scores": scores}))
        by_split = {}
        for split in ("train", "val", "test"):
            mask = (queries["split"].to_numpy() == split) & answerable
            if mask.any():
                by_split[split] = summarise([ranked[i] for i in np.flatnonzero(mask)],
                                            [gold[i] for i in np.flatnonzero(mask)])
        mask = answerable
        by_split["all"] = summarise([ranked[i] for i in np.flatnonzero(mask)], [gold[i] for i in np.flatnonzero(mask)])
        t = timing[method]
        metrics[method] = {"label": LABELS[method], "splits": by_split, "index_seconds": t["index_seconds"],
                           "ms_per_query": round(1000 * t["query_seconds"] / max(len(queries), 1), 2)}
        print(f"{LABELS[method]:<22} test hit@5={by_split.get('test', {}).get('hit@5')}  "
              f"MRR@10={by_split.get('test', {}).get('mrr@10')}  ({metrics[method]['ms_per_query']} ms/query)")

    info = {"documents": len(docs), "passages": len(passages), "queries": len(queries),
            "answerable_queries": int(answerable.sum()),
            "dense_model": getattr(encoder, "name", type(encoder).__name__),
            "reranker": getattr(reranker, "name", None) if reranker is not None else None,
            "settings": {k: getattr(config, k) for k in ("CHUNK_WORDS", "CHUNK_OVERLAP", "CANDIDATES", "RRF_K",
                                                          "RERANK_DEPTH", "RERANK_CHUNKS_PER_DOC")}}
    return {"info": info, "methods": metrics}, pd.concat(runs, ignore_index=True)


def plot_comparison(results: dict, path) -> None:
    """One panel per dataset: hit@5 on the test split with 95% bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE

    names = [n for n in results if results[n]["methods"]]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.2), squeeze=False, facecolor=SURFACE)
    for ax, name in zip(axes[0], names):
        methods = [m for m in METHODS if m in results[name]["methods"]]
        rows = [results[name]["methods"][m]["splits"].get("test") or results[name]["methods"][m]["splits"]["all"]
                for m in methods]
        values = [r["hit@5"] for r in rows]
        lo = [v - r["hit@5_ci95"][0] for v, r in zip(values, rows)]
        hi = [r["hit@5_ci95"][1] - v for v, r in zip(values, rows)]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, values, height=0.55, color=SERIES[0], zorder=2)
        ax.errorbar(values, y, xerr=[lo, hi], fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3, zorder=3)
        for yi, v, h in zip(y, values, hi):
            ax.text(v + h + 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [LABELS[m] for m in methods], color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        n = rows[0]["n_queries"]
        title = {"techqa": "TechQA (IBM Technotes)", "stackexchange": "Stack Exchange"}.get(name, name)
        ax.set_title(f"{title}\nhit@5, test set, {n:,} answerable queries", loc="left", fontsize=10, color=INK)
    fig.text(0.01, 0.01, "Share of questions whose correct document is in the top 5. Whiskers: 95% bootstrap interval.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dense-model", default=config.DENSE_MODEL)
    parser.add_argument("--reranker", default=config.RERANKER_MODEL, help="'none' to skip reranking")
    parser.add_argument("--max-queries-per-split", type=int, help="random sample per split (speeds up big datasets)")
    args = parser.parse_args(argv)

    from src.retrieval.retrievers import CrossEncoderReranker, SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(args.dense_model, query_prefix=config.DENSE_QUERY_PREFIX)
    reranker = None if args.reranker.lower() == "none" else CrossEncoderReranker(args.reranker)
    run_all(args.datasets, encoder, reranker, args.max_queries_per_split)


def run_all(datasets, encoder, reranker, max_per_split=None) -> dict:
    path = config.RESULTS_DIR / "retrieval_metrics.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    out_dir = config.DATA_DIR / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        results[name], runs = run_dataset(name, encoder, reranker, max_per_split)
        runs.to_parquet(out_dir / f"{name}_runs.parquet", index=False)
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    for name in datasets:
        results[name]["info"]["device"] = device
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    plot_comparison(results, config.RESULTS_DIR / "retrieval_comparison.png")
    return results


if __name__ == "__main__":
    main()
