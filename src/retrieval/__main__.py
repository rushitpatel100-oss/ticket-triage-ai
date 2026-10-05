"""Compare retrieval methods on the real knowledge bases.

Run on a GPU (e.g. Google Colab, see notebooks/retrieval_colab.ipynb) after `python -m src.sources`:
    python -m src.retrieval                          # TechQA and Stack Exchange
    python -m src.retrieval --datasets techqa --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base
    python -m src.retrieval --max-queries-per-split 2000

Methods:
    bm25                  keyword search
    dense                 embedding search (best passage per document)
    hybrid                reciprocal rank fusion of bm25 + dense
    hybrid+rerank:<name>  hybrid candidates re-scored by a cross-encoder

Scores are reported per split. "dev" (train + validation) is for choosing a method; "test" is kept for the
final number. Writes results/retrieval_metrics.json, results/retrieval_comparison.png and, for the
resolve-or-escalate model, data/processed/retrieval/<dataset>_runs.parquet (top results for every query).
Passage embeddings are cached in data/processed/retrieval/ so a rerun skips the slow encoding step.
"""

import argparse
import hashlib
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
BASE_METHODS = ("bm25", "dense", "hybrid")
LABELS = {"bm25": "BM25 (keywords)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (RRF)"}
RERANKER_SHORT = {"ms-marco-MiniLM-L-6-v2": "MiniLM", "bge-reranker-base": "BGE base",
                  "bge-reranker-v2-m3": "BGE v2-m3"}
SPLITS = {"train": ("train",), "val": ("val",), "dev": ("train", "val"), "test": ("test",)}


def rerank_key(model_name: str) -> str:
    return f"hybrid+rerank:{model_name.rstrip('/').split('/')[-1]}"


def label(method: str) -> str:
    if method in LABELS:
        return LABELS[method]
    short = method.split(":", 1)[1]
    return f"Hybrid + rerank ({RERANKER_SHORT.get(short, short)})"


def select_queries(queries: pd.DataFrame, max_per_split: int | None, seed: int = config.SEED) -> pd.DataFrame:
    if not max_per_split:
        return queries.reset_index(drop=True)
    parts = [g.sample(min(len(g), max_per_split), random_state=seed) for _, g in queries.groupby("split")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def cached_passage_embeddings(name: str, encoder, passage_texts: list[str]) -> np.ndarray:
    """Encode passages once per (dataset, model, passages); reuse the saved array afterwards."""
    model = getattr(encoder, "name", type(encoder).__name__).replace("/", "__")
    if getattr(encoder, "fp16", False):
        model += "_fp16"
    digest = hashlib.md5("\x00".join(passage_texts).encode("utf-8")).hexdigest()[:12]
    path = config.DATA_DIR / "retrieval" / f"{name}_{model}_{digest}.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(passage_texts):
            print(f"Using cached passage embeddings: {path.name}")
            return emb
    emb = encoder.encode_docs(passage_texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def run_dataset(name: str, encoder, rerankers=(), max_per_split: int | None = None) -> tuple[dict, pd.DataFrame]:
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
    emb = cached_passage_embeddings(name, encoder, passage_texts)
    dense = DenseRetriever(encoder, passage_texts, passages["doc_pos"].to_numpy(), starts, passage_emb=emb)
    index_s = time.time() - t
    t = time.time()
    query_emb = encoder.encode_queries(texts)
    rankings["dense"] = dense.search(query_emb, config.CANDIDATES)
    timing["dense"] = {"index_seconds": round(index_s, 1), "query_seconds": time.time() - t}

    t = time.time()
    rankings["hybrid"] = reciprocal_rank_fusion([rankings["bm25"], rankings["dense"]])
    timing["hybrid"] = {"index_seconds": timing["bm25"]["index_seconds"] + timing["dense"]["index_seconds"],
                        "query_seconds": timing["bm25"]["query_seconds"] + timing["dense"]["query_seconds"]
                        + time.time() - t}

    for reranker in rerankers:
        key = rerank_key(reranker.name)
        t = time.time()
        rankings[key] = rerank(reranker, texts, query_emb, rankings["hybrid"], dense, passage_texts,
                               config.RERANK_DEPTH, config.RERANK_CHUNKS_PER_DOC)
        timing[key] = {"index_seconds": timing["hybrid"]["index_seconds"],
                       "query_seconds": timing["hybrid"]["query_seconds"] + time.time() - t}

    doc_ids = docs["doc_id"].to_numpy()
    answerable = queries["answerable"].to_numpy()
    split = queries["split"].to_numpy()
    gold = queries["gold_doc_ids"].tolist()
    metrics, runs = {}, []
    for method, ranking in rankings.items():
        top = ranking.top(config.RUN_DEPTH)
        ranked = [[doc_ids[i] for i in row if i >= 0] for row in top.idx]
        scores = [[float(s) for s, i in zip(srow, irow) if i >= 0] for srow, irow in zip(top.scores, top.idx)]
        runs.append(pd.DataFrame({"query_id": queries["query_id"], "split": split, "answerable": answerable,
                                  "method": method, "doc_ids": ranked, "scores": scores}))
        by_split = {}
        for split_name, members in SPLITS.items():
            rows = np.flatnonzero(np.isin(split, members) & answerable)
            if len(rows):
                by_split[split_name] = summarise([ranked[i] for i in rows], [gold[i] for i in rows])
        t = timing[method]
        metrics[method] = {"label": label(method), "splits": by_split, "index_seconds": t["index_seconds"],
                           "ms_per_query": round(1000 * t["query_seconds"] / max(len(queries), 1), 2)}
        dev, test = by_split.get("dev", {}), by_split.get("test", {})
        print(f"{label(method):<30} dev hit@5={dev.get('hit@5')} MRR@10={dev.get('mrr@10')} | "
              f"test hit@5={test.get('hit@5')} MRR@10={test.get('mrr@10')} ({metrics[method]['ms_per_query']} ms/query)")

    info = {"documents": len(docs), "passages": len(passages), "queries": len(queries),
            "answerable_queries": int(answerable.sum()),
            "dense_model": getattr(encoder, "name", type(encoder).__name__),
            "rerankers": [r.name for r in rerankers],
            "fp16": bool(getattr(encoder, "fp16", False)) or any(getattr(r, "fp16", False) for r in rerankers),
            "settings": {k: getattr(config, k) for k in ("CHUNK_WORDS", "CHUNK_OVERLAP", "CANDIDATES", "RRF_K",
                                                          "RERANK_DEPTH", "RERANK_CHUNKS_PER_DOC")}}
    return {"info": info, "methods": metrics}, pd.concat(runs, ignore_index=True)


def plot_comparison(results: dict, path) -> None:
    """One panel per dataset: hit@5 on the test split with 95% bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE

    names = [n for n in results if results[n].get("methods")]
    n_methods = max(len(results[n]["methods"]) for n in names)
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 0.62 * n_methods + 1.4), squeeze=False,
                             facecolor=SURFACE)
    for ax, name in zip(axes[0], names):
        methods = list(results[name]["methods"])
        rows = [results[name]["methods"][m]["splits"].get("test") or results[name]["methods"][m]["splits"]["dev"]
                for m in methods]
        values = [r["hit@5"] for r in rows]
        lo = [v - r["hit@5_ci95"][0] for v, r in zip(values, rows)]
        hi = [r["hit@5_ci95"][1] - v for v, r in zip(values, rows)]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, values, height=0.5, color=SERIES[0], zorder=2)
        ax.errorbar(values, y, xerr=[lo, hi], fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3, zorder=3)
        for yi, v, h in zip(y, values, hi):
            ax.text(v + h + 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [results[name]["methods"][m]["label"] for m in methods], color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        title = {"techqa": "TechQA (IBM Technotes)", "stackexchange": "Stack Exchange"}.get(name, name)
        ax.set_title(f"{title}\nhit@5, test set, {rows[0]['n_queries']:,} answerable questions", loc="left",
                     fontsize=10, color=INK)
    fig.text(0.01, 0.01, "Share of questions whose correct document is in the top 5. Whiskers: 95% bootstrap interval.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run_all(datasets, encoder, rerankers=(), max_per_split=None) -> dict:
    path = config.RESULTS_DIR / "retrieval_metrics.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    out_dir = config.DATA_DIR / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    for name in datasets:
        results[name], runs = run_dataset(name, encoder, rerankers, max_per_split)
        results[name]["info"]["device"] = device
        runs.to_parquet(out_dir / f"{name}_runs.parquet", index=False)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    plot_comparison(results, config.RESULTS_DIR / "retrieval_comparison.png")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dense-model", default=config.DENSE_MODEL)
    parser.add_argument("--rerankers", nargs="+", default=[config.RERANKER_MODEL], help="'none' to skip reranking")
    parser.add_argument("--fp16", action="store_true", help="run the models in half precision on a GPU")
    parser.add_argument("--max-queries-per-split", type=int, help="random sample per split (speeds up big datasets)")
    args = parser.parse_args(argv)

    from src.retrieval.retrievers import CrossEncoderReranker, SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(args.dense_model, query_prefix=config.DENSE_QUERY_PREFIX, fp16=args.fp16)
    names = [r for r in args.rerankers if r.lower() != "none"]
    rerankers = [CrossEncoderReranker(r, fp16=args.fp16) for r in names]
    run_all(args.datasets, encoder, rerankers, args.max_queries_per_split)


if __name__ == "__main__":
    main()
