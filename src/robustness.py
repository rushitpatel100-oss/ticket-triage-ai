"""E5: how much do messy tickets hurt, and how much does the intake clean-up win back?

Real tickets are not clean forum questions. They arrive with e-mail signatures and disclaimers, forwarded
threads, typos, or as a single line ("VPN broken"). Each test question gets those problems added, and the
same models are scored again:

- retrieval (BM25, embeddings, hybrid) on TechQA and Stack Exchange: hit@5 and MRR@10,
- routing (TF-IDF and fine-tuned DistilBERT) on the synthetic ticket test set: macro-F1,

each with and without `src.intake.clean_ticket` in front.

    python -m src.robustness                        # all of it (GPU recommended for the embeddings)
    python -m src.robustness --skip-routing --se-sample 300

Writes results/robustness_metrics.json and results/robustness.png.
"""

import argparse
import json
import random

import numpy as np

from src import config
from src.intake import clean_ticket

SIGNATURES = [
    "Kind regards,\nAlex Morgan\nFinance Analyst | Example Holdings Ltd\nT: +44 20 7946 0000\n\n"
    "This email and any attachments are confidential and intended solely for the addressee. If you have received "
    "this email in error please notify the sender and delete it.",
    "Thanks,\nSam\n\nSent from my iPhone",
    "Best regards,\n\nJordan Lee\nOperations Coordinator\nExample Group plc\nPlease consider the environment before "
    "printing this email.\nDISCLAIMER: the information in this message is confidential and may be legally privileged.",
]
NOISE = ("signature", "forwarded", "typos", "short", "messy")
VARIANTS = ("clean", *NOISE, "clean+cleanup", "signature+cleanup", "forwarded+cleanup", "messy+cleanup")


def add_signature(text: str, rng: random.Random) -> str:
    return f"{text}\n\n{rng.choice(SIGNATURES)}"


def add_forwarded(text: str, other: str, rng: random.Random) -> str:
    """The ticket on top of an unrelated older message, as when someone replies to an old thread."""
    subject = other.split("\n", 1)[0][:80]
    quoted = "\n".join(f"> {line}" if rng.random() < 0.5 else line for line in other.split("\n"))
    return (f"{text}\n\n-----Original Message-----\nFrom: Service Desk <servicedesk@example.com>\n"
            f"Sent: Monday, 5 October 2026 09:12\nTo: Alex Morgan <alex.morgan@example.com>\n"
            f"Subject: RE: {subject}\n\n{quoted}")


def add_typos(text: str, rng: random.Random, rate: float = 0.12) -> str:
    """Swap, drop or double a letter in about `rate` of the longer words."""
    def typo(word: str) -> str:
        if len(word) < 4 or not word.isalpha() or rng.random() >= rate:
            return word
        i = rng.randrange(1, len(word) - 1)
        op = rng.randrange(3)
        if op == 0:
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if op == 1:
            return word[:i] + word[i + 1:]
        return word[:i] + word[i] + word[i:]
    return "\n".join(" ".join(typo(w) for w in line.split(" ")) for line in text.split("\n"))


def make_short(text: str, max_words: int = 12) -> str:
    """Only the first line (the subject or title), at most max_words words: a terse ticket."""
    first = next((line for line in text.split("\n") if line.strip()), text)
    return " ".join(first.split()[:max_words])


def make_variants(texts: list[str], seed: int = config.SEED) -> dict[str, list[str]]:
    """Every noise type for every text. The forwarded thread is another, unrelated text from the same set."""
    texts = [str(t) for t in texts]
    rng = random.Random(seed)
    others = [texts[(i + 1 + rng.randrange(max(len(texts) - 1, 1))) % len(texts)] for i in range(len(texts))]

    def per_text(fn):
        return [fn(t, random.Random(f"{seed}:{i}")) for i, t in enumerate(texts)]
    out = {"clean": texts,
           "signature": per_text(add_signature),
           "forwarded": [add_forwarded(t, o, random.Random(f"{seed}:{i}")) for i, (t, o) in enumerate(zip(texts, others))],
           "typos": per_text(add_typos),
           "short": [make_short(t) for t in texts]}
    out["messy"] = [add_typos(add_signature(add_forwarded(t, o, random.Random(f"{seed}:f{i}")),
                                            random.Random(f"{seed}:s{i}")), random.Random(f"{seed}:t{i}"))
                    for i, (t, o) in enumerate(zip(texts, others))]
    for name in ("clean", "signature", "forwarded", "messy"):
        out[f"{name}+cleanup"] = [clean_ticket(t) for t in out[name]]
    return out


def retrieval_robustness(name: str, encoder, sample: int | None = None, seed: int = config.SEED) -> dict:
    """hit@5 and MRR@10 on the answerable test questions (and dev for TechQA, which is small), per variant."""
    from src.retrieval.__main__ import cached_passage_embeddings
    from src.retrieval.core import chunk_documents, doc_starts, reciprocal_rank_fusion
    from src.retrieval.metrics import summarise
    from src.retrieval.retrievers import BM25, DenseRetriever
    from src.sources.common import load_tables

    docs, queries = load_tables(name)
    docs = docs.reset_index(drop=True)
    q = queries[queries["answerable"]]
    if name != "techqa":
        q = q[q["split"] == "test"]
    if sample and len(q) > sample:
        q = q.sample(sample, random_state=seed)
    q = q.reset_index(drop=True)
    passages = chunk_documents(docs)
    starts = doc_starts(passages["doc_pos"].to_numpy(), len(docs))
    passage_texts = passages["text"].tolist()
    bm25 = BM25().fit(docs["text"].tolist())
    emb = cached_passage_embeddings(name, encoder, passage_texts)
    dense = DenseRetriever(encoder, passage_texts, passages["doc_pos"].to_numpy(), starts, passage_emb=emb)
    doc_ids = docs["doc_id"].to_numpy()
    gold = q["gold_doc_ids"].tolist()

    def ranked(r):
        top = r.top(10)
        return [[doc_ids[i] for i in row if i >= 0] for row in top.idx]

    out = {"questions": len(q), "variants": {}}
    for variant, texts in make_variants(q["text"].tolist(), seed).items():
        r_bm25 = bm25.search(texts, config.CANDIDATES)
        r_dense = dense.search(encoder.encode_queries(texts), config.CANDIDATES)
        r_hybrid = reciprocal_rank_fusion([r_bm25, r_dense])
        out["variants"][variant] = {
            "words_median": int(np.median([len(t.split()) for t in texts])),
            **{m: {k: s[k] for k in ("hit@5", "mrr@10")} for m, s in
               (("bm25", summarise(ranked(r_bm25), gold)), ("dense", summarise(ranked(r_dense), gold)),
                ("hybrid", summarise(ranked(r_hybrid), gold)))}}
        v = out["variants"][variant]
        print(f"{name} {variant:<18} MRR@10 bm25 {v['bm25']['mrr@10']:.3f}  dense {v['dense']['mrr@10']:.3f}  "
              f"hybrid {v['hybrid']['mrr@10']:.3f}")
    return out


def predict_labels(entry: dict, texts: list[str], batch_size: int = 64) -> list[str]:
    """Top label for many texts at once (the demo's TicketClassifier scores one ticket at a time)."""
    if entry["kind"] != "transformer":
        return list(entry["model"].predict(texts))
    import torch

    model, tok = entry["model"], entry["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=config.MAX_LENGTH, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model(**enc).logits.argmax(-1).tolist()
        out += [model.config.id2label[j] for j in ids]
    return out


def routing_robustness(tasks=tuple(config.TASKS), seed: int = config.SEED, classifier=None) -> dict:
    """Macro-F1 of each saved routing model (TF-IDF baseline and, if present, DistilBERT) per variant."""
    from sklearn.metrics import f1_score

    from src.data import load_splits
    from src.predict import TicketClassifier

    _, _, test = load_splits()
    variants = make_variants(test["text"].tolist(), seed)
    out = {"tickets": len(test), "tasks": {}}
    for task in tasks:
        if config.TASKS[task] not in test:
            continue
        clf = classifier or TicketClassifier(tasks=[task])
        entries = {}
        base = config.MODELS_DIR / f"baseline_{task}.joblib"
        if base.exists():
            import joblib
            entries["TF-IDF + LogReg"] = {"kind": "baseline", "name": "TF-IDF", "model": joblib.load(base)["model"]}
        if task in clf.models and clf.models[task]["kind"] == "transformer":
            entries["DistilBERT"] = clf.models[task]
        y = test[config.TASKS[task]].to_numpy()
        for model_name, entry in entries.items():
            row = {}
            for variant, texts in variants.items():
                preds = predict_labels(entry, texts)
                row[variant] = round(float(f1_score(y, preds, average="macro", zero_division=0)), 4)
            out["tasks"].setdefault(task, {})[model_name] = row
            print(f"routing {task} {model_name}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


def run_all(datasets=("techqa", "stackexchange"), encoder=None, routing: bool = True, se_sample: int = 1000) -> dict:
    results = {"variants": list(VARIANTS), "retrieval": {}, "routing": None}
    if datasets:
        if encoder is None:
            import torch

            from src.retrieval.retrievers import SentenceTransformerEncoder

            encoder = SentenceTransformerEncoder(config.DENSE_MODEL, query_prefix=config.DENSE_QUERY_PREFIX,
                                                 fp16=torch.cuda.is_available())
        for name in datasets:
            results["retrieval"][name] = retrieval_robustness(name, encoder, None if name == "techqa" else se_sample)
    if routing:
        results["routing"] = routing_robustness()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "robustness_metrics.json").write_text(json.dumps(results, indent=2))
    plot_robustness(results, config.RESULTS_DIR / "robustness.png")
    return results


def plot_robustness(results: dict, path) -> None:
    """Score as a share of the clean score, per noise type, before and after clean-up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE

    panels = []
    for name, res in results.get("retrieval", {}).items():
        v = res["variants"]
        title = {"techqa": "Retrieval, TechQA", "stackexchange": "Retrieval, Stack Exchange"}.get(name, name)
        panels.append((f"{title} (embeddings, MRR@10)", {k: v[k]["dense"]["mrr@10"] for k in v}))
    for task, models in (results.get("routing") or {}).get("tasks", {}).items():
        for model, row in models.items():
            if task == "queue":
                panels.append((f"Routing to a team ({model}, macro-F1)", row))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), squeeze=False, facecolor=SURFACE)
    for ax, (title, row) in zip(axes[0], panels):
        clean = row["clean"] or 1e-9
        kinds = [k for k in ("signature", "forwarded", "messy") if k in row] + [k for k in ("typos", "short") if k in row]
        x = np.arange(len(kinds))
        raw = [row[k] / clean for k in kinds]
        fixed = [row.get(f"{k}+cleanup", np.nan) / clean for k in kinds]
        ax.bar(x - 0.19, raw, 0.36, color=SERIES[1], label="As received", zorder=3)
        ax.bar(x + 0.19, fixed, 0.36, color=SERIES[0], label="After clean-up", zorder=3)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xticks(x, kinds, fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_ylabel("Share of the clean score", fontsize=8, color=INK_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    axes[0][0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.01, "Clean-up removes signatures, disclaimers and quoted threads; it cannot fix typos or add "
             "missing detail, so those have no second bar.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=["techqa", "stackexchange"])
    parser.add_argument("--skip-routing", action="store_true")
    parser.add_argument("--se-sample", type=int, default=1000, help="Stack Exchange test questions to use")
    args = parser.parse_args(argv)
    run_all(args.datasets, routing=not args.skip_routing, se_sample=args.se_sample)


if __name__ == "__main__":
    main()
