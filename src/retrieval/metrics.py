"""Retrieval metrics. Only answerable queries count: unanswerable ones have no correct document."""

import numpy as np

KS = (1, 5, 10)


def per_query(ranked: list[list[str]], gold: list[list[str]], cutoff: int = 10) -> dict[str, np.ndarray]:
    """Per-query values: rank of the first correct document, reciprocal rank, nDCG and recall at `cutoff`."""
    first, rr, ndcg, recall = [], [], [], []
    for docs, relevant in zip(ranked, gold):
        relevant = set(relevant)
        hits = [i for i, d in enumerate(docs[:cutoff]) if d in relevant]
        all_hits = [i for i, d in enumerate(docs) if d in relevant]
        first.append(all_hits[0] + 1 if all_hits else np.inf)
        rr.append(1.0 / (hits[0] + 1) if hits else 0.0)
        dcg = sum(1.0 / np.log2(i + 2) for i in hits)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), cutoff)))
        ndcg.append(dcg / ideal if ideal else 0.0)
        recall.append(len(hits) / len(relevant) if relevant else 0.0)
    return {"first_rank": np.array(first), "rr": np.array(rr), "ndcg": np.array(ndcg), "recall": np.array(recall)}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean, resampling queries."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def summarise(ranked: list[list[str]], gold: list[list[str]]) -> dict:
    v = per_query(ranked, gold)
    out = {"n_queries": len(gold)}
    for k in KS:
        out[f"hit@{k}"] = round(float(np.mean(v["first_rank"] <= k)), 4) if len(gold) else None
    out["mrr@10"] = round(float(v["rr"].mean()), 4) if len(gold) else None
    out["ndcg@10"] = round(float(v["ndcg"].mean()), 4) if len(gold) else None
    out["recall@10"] = round(float(v["recall"].mean()), 4) if len(gold) else None
    out["hit@5_ci95"] = bootstrap_ci((v["first_rank"] <= 5).astype(float))
    out["mrr@10_ci95"] = bootstrap_ci(v["rr"])
    return out
