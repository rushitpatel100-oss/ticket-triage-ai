"""Write the retrieval results table into the README (between RETRIEVAL markers).

Run:  python -m src.retrieval.report
"""

import json

from src import config

START, END = "<!-- RETRIEVAL:START -->", "<!-- RETRIEVAL:END -->"
TITLES = {"techqa": "TechQA (IBM Technotes)", "stackexchange": "Stack Exchange (Super User, Ask Ubuntu)"}


def fmt(x) -> str:
    """Two decimals, rounding halves up (0.625 -> 0.63), as people expect in a table."""
    from decimal import ROUND_HALF_UP, Decimal

    return "–" if x is None else str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def table(results: dict, paired: dict | None = None) -> str:
    lines = []
    for name, res in results.items():
        info = res["info"]
        test_n = next(iter(res["methods"].values()))["splits"].get("test", {}).get("n_queries", 0)
        dev_n = next(iter(res["methods"].values()))["splits"].get("dev", {}).get("n_queries", 0)
        lines += [f"**{TITLES.get(name, name)}**: {info['documents']:,} documents ({info['passages']:,} passages); "
                  f"{dev_n:,} dev and {test_n:,} test questions that have an answer.", "",
                  "| Method | Dev hit@5 | Dev MRR@10 | Test hit@5 | Test MRR@10 | Test hit@5 95% CI | ms per query |",
                  "|---|---|---|---|---|---|---|"]
        best_dev = max(res["methods"].values(), key=lambda m: m["splits"]["dev"]["mrr@10"])["label"]
        for m in res["methods"].values():
            dev, test = m["splits"].get("dev", {}), m["splits"].get("test", {})
            ci = test.get("hit@5_ci95") or [None, None]
            name_cell = f"**{m['label']}**" if m["label"] == best_dev else m["label"]
            lines.append(f"| {name_cell} | {fmt(dev.get('hit@5'))} | {fmt(dev.get('mrr@10'))} | {fmt(test.get('hit@5'))} "
                         f"| {fmt(test.get('mrr@10'))} | {fmt(ci[0])}–{fmt(ci[1])} | {m['ms_per_query']:.0f} |")
        lines.append("")
        if paired and name in paired:
            ref = paired[name]["reference"]
            lines.append(f"Paired difference in dev MRR@10 against {ref} (bootstrap over questions, 95% interval):")
            lines.append("")
            for method, splits in paired[name]["methods"].items():
                d = splits.get("dev", {}).get("mrr@10")
                if d:
                    label = res["methods"][method]["label"] if method in res["methods"] else method
                    verdict = "better" if d["ci95"][0] > 0 else "worse" if d["ci95"][1] < 0 else "no clear difference"
                    lines.append(f"- {label}: {d['mean_diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}), "
                                 f"{verdict}")
            lines.append("")
    lines.append(f"Models: `{info['dense_model']}` embeddings; rerankers "
                 + ", ".join(f"`{r}`" for r in info["rerankers"]) + f". Timed on a {info.get('device', 'GPU')}"
                 + (" in half precision." if info.get("fp16") else "."))
    return "\n".join(lines)


def main() -> None:
    results = json.loads((config.RESULTS_DIR / "retrieval_metrics.json").read_text())
    paired_path = config.RESULTS_DIR / "retrieval_paired.json"
    paired = json.loads(paired_path.read_text()) if paired_path.exists() else None
    readme = config.ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"README.md needs {START} and {END} markers")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{table(results, paired)}\n{END}{after}")
    print(table(results, paired))


if __name__ == "__main__":
    main()
