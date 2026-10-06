"""Run the answer step on the real knowledge bases.

Run on a GPU after `python -m src.retrieval` (see notebooks/answer_colab.ipynb):
    python -m src.answer                                   # open model, both datasets
    python -m src.answer --datasets techqa --max-answers 100
    python -m src.answer --backend claude                  # needs ANTHROPIC_API_KEY

For every question with retrieval results:
1. Evidence: the passage of each of the top CONTEXT_DOCS articles (embedding search) most similar to the ticket.
2. Judge: the probability the model gives to "Yes" when asked whether the articles contain the answer.
3. Answers (test questions, up to --max-answers per dataset): a drafted reply with citations, or NOT_FOUND.
4. Red team: prompt-injection cases (src/answer/redteam.py).

Writes data/processed/answer/<dataset>_judge.parquet (the new signal for `python -m src.decision --llm-judge`),
data/processed/answer/<dataset>_answers.parquet and results/answer_metrics.json.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.answer import redteam
from src.answer.metrics import answer_report
from src.decision.model import scores
from src.retrieval.__main__ import cached_passage_embeddings
from src.retrieval.core import chunk_documents, doc_starts
from src.retrieval.retrievers import DenseRetriever
from src.sources.common import load_tables

DATASETS = ("techqa", "stackexchange")
SPLITS = {"dev": ("train", "val"), "test": ("test",)}


def build_evidence(name: str, encoder, k: int = config.CONTEXT_DOCS) -> pd.DataFrame:
    """For each question: its top-k articles from embedding search, each as its best-matching passage."""
    docs, queries = load_tables(name)
    docs = docs.reset_index(drop=True)
    runs = pd.read_parquet(config.DATA_DIR / "retrieval" / f"{name}_runs.parquet")
    dense = runs[runs["method"] == config.PRIMARY_RETRIEVER].set_index("query_id")
    passages = chunk_documents(docs)
    starts = doc_starts(passages["doc_pos"].to_numpy(), len(docs))
    texts = passages["text"].tolist()
    emb = cached_passage_embeddings(name, encoder, texts)
    retriever = DenseRetriever(encoder, texts, passages["doc_pos"].to_numpy(), starts, passage_emb=emb)
    position = {d: i for i, d in enumerate(docs["doc_id"])}
    q = queries.set_index("query_id").loc[dense.index]
    q_emb = encoder.encode_queries(q["text"].tolist())

    rows = []
    for i, qid in enumerate(dense.index):
        top = list(dense.at[qid, "doc_ids"])[:k]
        evidence = []
        for doc_id in top:
            p = int(retriever.best_passages(q_emb[i], position[doc_id], 1)[0])
            title = docs.at[position[doc_id], "title"] or ""
            text = texts[p]
            if title and text.startswith(title + "\n"):  # later passages repeat the title; keep it once
                text = text[len(title) + 1:]
            evidence.append({"doc_id": doc_id, "title": title, "text": text})
        gold = set(q.at[qid, "gold_doc_ids"])
        answerable = bool(q.at[qid, "answerable"])
        rows.append({"query_id": qid, "split": q.at[qid, "split"], "text": q.at[qid, "text"],
                     "answerable": answerable, "resolvable": answerable and bool(gold & set(top)),
                     "gold_doc_ids": list(gold), "gold_answer": q.at[qid, "gold_answer"], "evidence": evidence})
    return pd.DataFrame(rows)


def judge_report(table: pd.DataFrame) -> dict:
    out = {}
    for split, members in SPLITS.items():
        part = table[table["split"].isin(members)]
        if len(part):
            out[split] = {"resolvable": scores(part["llm_p_yes"].to_numpy(), part["resolvable"].to_numpy()),
                          "answerable": scores(part["llm_p_yes"].to_numpy(), part["answerable"].to_numpy())}
    return out


def run_dataset(name: str, llm, encoder, max_answers: int, seed: int = config.SEED, limit: int | None = None) -> dict:
    out_dir = config.DATA_DIR / "answer"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_evidence(name, encoder)
    if limit and len(table) > limit:  # smoke test: a small random sample
        table = table.sample(limit, random_state=seed).reset_index(drop=True)
    print(f"\n=== {name}: judging {len(table):,} questions ===")

    t = time.time()
    p_yes, mass = llm.judge(table["text"].tolist(), table["evidence"].tolist())
    judge_s = time.time() - t
    bad = int(np.isnan(p_yes).sum())
    if bad:  # e.g. half-precision overflow; treat as "unsure" and report it
        print(f"warning: {bad} judgements were NaN; set to 0.5")
        p_yes = np.nan_to_num(p_yes, nan=0.5)
    table["llm_p_yes"], table["llm_yes_no_mass"] = p_yes, mass
    table[["query_id", "split", "answerable", "resolvable", "llm_p_yes", "llm_yes_no_mass"]].to_parquet(
        out_dir / f"{name}_judge.parquet", index=False)
    report = {"questions": len(table), "judge_nan": bad,
              "judge_ms_per_question": round(1000 * judge_s / max(len(table), 1), 1),
              "yes_no_mass_median": round(float(np.median(mass)), 4), "judge": judge_report(table)}

    test = table[table["split"] == "test"]
    if max_answers and len(test) > max_answers:
        test = test.sample(max_answers, random_state=seed)
    if len(test):
        t = time.time()
        answers = llm.answer(test["text"].tolist(), test["evidence"].tolist())
        answer_s = time.time() - t
        saved = test[["query_id", "resolvable", "gold_answer"]].copy()
        saved["answer"] = answers
        saved["evidence_ids"] = [[e["doc_id"] for e in ev] for ev in test["evidence"]]
        saved["llm_p_yes"] = test["llm_p_yes"].to_numpy()
        saved.to_parquet(out_dir / f"{name}_answers.parquet", index=False)
        report["answers"] = answer_report(answers, saved["evidence_ids"].tolist(), test["gold_doc_ids"].tolist(),
                                          test["resolvable"].to_numpy(), test["gold_answer"].tolist())
        report["answers"]["seconds_per_answer"] = round(answer_s / len(test), 2)
        report["answer_examples"] = [{"question": q[:200], "resolvable": bool(r), "answer": a[:500]}
                                     for q, r, a in zip(test["text"].head(3), test["resolvable"].head(3), answers[:3])]
    jr = report["judge"].get("test", {}).get("resolvable", {})
    print(f"{name}: judge test AUROC (resolvable) {jr.get('auroc')}, {report['judge_ms_per_question']} ms/question")
    return report


def run_redteam(llm) -> dict:
    cases = redteam.cases()
    replies = llm.answer([c["ticket"] for c in cases], [c["evidence"] for c in cases])
    out = redteam.score(replies, [c["kind"] for c in cases])
    out["examples"] = [{"kind": c["kind"], "ticket": c["ticket"][:120], "reply": r[:200]}
                       for c, r in zip(cases[:2] + cases[-2:], replies[:2] + replies[-2:])]
    print(f"red team: {out['attack_success_rate']:.0%} of injection attempts succeeded")
    return out


def run_all(datasets, llm, encoder, max_answers: int, limit: int | None = None, redteam_too: bool = True) -> dict:
    """Results are merged into results/answer_metrics.json, so datasets can be run one at a time."""
    path = config.RESULTS_DIR / "answer_metrics.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results.update({"backend": type(llm).__name__, "model": getattr(llm, "name", None), "limit": limit})
    for name in datasets:
        if (config.DATA_DIR / "retrieval" / f"{name}_runs.parquet").exists():
            results[name] = run_dataset(name, llm, encoder, max_answers, limit=limit)
    if redteam_too:
        results["redteam"] = run_redteam(llm)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backend", choices=("open", "claude"), default="open")
    parser.add_argument("--model", help="model name (default: config.OPEN_LLM or config.CLAUDE_MODEL)")
    parser.add_argument("--max-answers", type=int, default=150, help="test questions per dataset to draft answers for")
    parser.add_argument("--batch-size", type=int, default=16, help="judge batch size (answers use half)")
    parser.add_argument("--skip-redteam", action="store_true")
    parser.add_argument("--limit", type=int, help="judge only this many random questions per dataset (smoke test)")
    args = parser.parse_args(argv)

    from src.answer.llm import ClaudeModel, OpenModel
    from src.retrieval.retrievers import SentenceTransformerEncoder

    llm = (OpenModel(args.model or config.OPEN_LLM, batch_size=args.batch_size) if args.backend == "open"
           else ClaudeModel(args.model or config.CLAUDE_MODEL))
    encoder = SentenceTransformerEncoder(config.DENSE_MODEL, query_prefix=config.DENSE_QUERY_PREFIX, fp16=True)
    results = run_all(args.datasets, llm, encoder, args.max_answers, limit=args.limit, redteam_too=not args.skip_redteam)
    print("ANSWER_START")
    print(json.dumps(results))
    print("ANSWER_END")


if __name__ == "__main__":
    main()
