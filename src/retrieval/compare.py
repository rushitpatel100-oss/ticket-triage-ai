"""Paired comparison of retrieval methods: is a difference real or noise?

Each method is scored on the same questions, so we bootstrap the per-question difference (resampling
questions) instead of comparing two separate intervals, which is far more sensitive.

Run after `python -m src.retrieval`:
    python -m src.retrieval.compare                    # every method against "hybrid"
    python -m src.retrieval.compare --reference dense
Writes results/retrieval_paired.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.retrieval.metrics import per_query
from src.sources.common import load_tables

SPLITS = {"dev": ("train", "val"), "test": ("test",)}


def paired_bootstrap(diff: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {"mean_diff": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "share_of_resamples_better": round(float((means > 0).mean()), 4),
            "questions_better": int((diff > 0).sum()), "questions_worse": int((diff < 0).sum())}


def compare(name: str, reference: str = "hybrid") -> dict:
    _, queries = load_tables(name)
    gold = dict(zip(queries["query_id"], queries["gold_doc_ids"]))
    runs = pd.read_parquet(config.DATA_DIR / "retrieval" / f"{name}_runs.parquet")
    runs = runs[runs["answerable"]]
    if reference not in set(runs["method"]):
        raise ValueError(f"{reference} not in runs: {sorted(set(runs['method']))}")

    values = {}
    for method, part in runs.groupby("method"):
        part = part.sort_values("query_id")
        v = per_query([list(d) for d in part["doc_ids"]], [gold[q] for q in part["query_id"]])
        values[method] = pd.DataFrame({"query_id": part["query_id"].to_numpy(), "split": part["split"].to_numpy(),
                                       "rr": v["rr"], "hit5": (v["first_rank"] <= 5).astype(float)})

    ref = values[reference].set_index("query_id")
    out = {}
    for method, table in values.items():
        if method == reference:
            continue
        table = table.set_index("query_id").loc[ref.index]
        out[method] = {}
        for split_name, members in SPLITS.items():
            mask = ref["split"].isin(members).to_numpy()
            if mask.any():
                out[method][split_name] = {
                    "mrr@10": paired_bootstrap(table["rr"].to_numpy()[mask] - ref["rr"].to_numpy()[mask]),
                    "hit@5": paired_bootstrap(table["hit5"].to_numpy()[mask] - ref["hit5"].to_numpy()[mask]),
                    "n_queries": int(mask.sum()),
                }
    return {"reference": reference, "methods": out}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["techqa", "stackexchange"])
    parser.add_argument("--reference", default="hybrid")
    args = parser.parse_args(argv)
    results = {}
    for name in args.datasets:
        if (config.DATA_DIR / "retrieval" / f"{name}_runs.parquet").exists():
            results[name] = compare(name, args.reference)
            for method, splits in results[name]["methods"].items():
                for split_name, r in splits.items():
                    m = r["mrr@10"]
                    print(f"{name:<14} {method:<40} {split_name:<4} MRR@10 diff vs {args.reference}: "
                          f"{m['mean_diff']:+.4f} CI {m['ci95']}  P(better)={m['share_of_resamples_better']}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "retrieval_paired.json").write_text(json.dumps(results, indent=2))
    print("PAIRED_START")
    print(json.dumps(results))
    print("PAIRED_END")


if __name__ == "__main__":
    main()
