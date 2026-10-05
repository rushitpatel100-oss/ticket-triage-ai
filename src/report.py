"""Turn results/metrics.json into a results table + chart and paste the table into README.md.

Run after training:  python -m src.report
"""

import json
from pathlib import Path

from src import config
from src.evaluate import plot_model_comparison

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def results_table(results: dict) -> str:
    rows = ["| Task | Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---|---|---|---|"]
    for task, models in results.items():
        best = max(m["macro_f1"] for m in models.values())
        for name, m in models.items():
            mark = "**" if m["macro_f1"] == best else ""
            rows.append(f"| {task} | {name} | {m['accuracy']:.3f} | {mark}{m['macro_f1']:.3f}{mark} | {m['weighted_f1']:.3f} |")
    n_test = next(iter(next(iter(results.values())).values()))["n_test"]
    rows.append(f"\n*Scores on {n_test:,} held-out test tickets. Best macro F1 per task in bold.*")
    return "\n".join(rows)


def main() -> None:
    metrics_file = config.RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise SystemExit("No results yet. Run `python -m src.baseline` (and `python -m src.train`) first.")
    results = json.loads(metrics_file.read_text())

    table = results_table(results)
    plot_model_comparison(results, config.RESULTS_DIR / "model_comparison.png")
    print(table)

    # Charts shown in the README (paths relative to the repo root)
    images = ["results/model_comparison.png"]
    for task in results:
        for kind in ("transformer", "tfidf"):  # show the transformer's matrix if it exists
            if (config.RESULTS_DIR / f"confusion_{task}_{kind}.png").exists():
                images.append(f"results/confusion_{task}_{kind}.png")
                break
    block = table + "\n\n" + "\n\n".join(f"![{Path(i).stem.replace('_', ' ')}]({i})" for i in images)

    readme = config.ROOT / "README.md"
    text = readme.read_text()
    if START in text and END in text:
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1]
        readme.write_text(f"{before}{START}\n{block}\n{END}{after}")
        print("\nUpdated the results table in README.md")


if __name__ == "__main__":
    main()
