"""Train and evaluate the resolve / assist / escalate decision on the real knowledge bases.

Run after `python -m src.retrieval` (it reads data/processed/retrieval/<dataset>_runs.parquet):
    python -m src.decision
    python -m src.decision --datasets techqa
    python -m src.decision --llm-judge      # add the LLM judgement from `python -m src.answer` as a signal

For each dataset:
1. Features and labels per question (src/decision/features.py). Label: "resolvable" = the correct article
   is among the top CONTEXT_DOCS that embedding search returns.
2. Three models (one-signal baseline, logistic regression, gradient boosting) scored with cross-fitted
   predictions on dev (train + validation); the better of logistic / boosting by average precision is selected.
3. Thresholds from the dev predictions: auto-resolve where the wrong-automation rate is <= TARGET_RISK with
   RISK_CONFIDENCE; escalate below the score that keeps ASSIST_RECALL of resolvable tickets.
4. Final models fitted on all of dev and evaluated once on test, including hard rules.
5. Transfer: the Stack Exchange model and thresholds applied unchanged to TechQA (a new knowledge base).

Writes results/decision_metrics.json, results/decision_risk_coverage.png and
data/processed/decision/<dataset>_predictions.parquet.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config
from src.decision.features import build_features, feature_columns
from src.decision.model import (
    apply_thresholds,
    candidate_models,
    columns_for,
    cross_fit,
    importance,
    recall_threshold,
    risk_controlled_threshold,
    risk_coverage_curve,
    scores,
)
from src.decision.policy import decide, rule_hits
from src.sources.common import load_tables

DATASETS = ("techqa", "stackexchange")
DEV = ("train", "val")
OTHER_TARGETS = (0.05, 0.20)
LABELS = {"score_only": "Best-match score only", "llm_only": "LLM judgement only", "logistic": "Logistic regression",
          "boosting": "Gradient boosting"}


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, queries = load_tables(name)
    runs = pd.read_parquet(config.DATA_DIR / "retrieval" / f"{name}_runs.parquet")
    return runs, queries


def thresholds_from(oof: np.ndarray, y: np.ndarray, target: float = config.TARGET_RISK) -> tuple[dict, float]:
    auto = risk_controlled_threshold(oof, y, target)
    escalate = min(recall_threshold(oof, y), auto["threshold"])
    return auto, escalate


def load_llm_judge(name: str) -> pd.DataFrame | None:
    path = config.DATA_DIR / "answer" / f"{name}_judge.parquet"
    return pd.read_parquet(path) if path.exists() else None


def evaluate_dataset(name: str, runs: pd.DataFrame, queries: pd.DataFrame,
                     extra: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, dict]:
    feats = build_features(runs, queries, extra=extra)
    cols = feature_columns(feats)
    dev, test = feats[feats["split"].isin(DEV)].reset_index(drop=True), feats[feats["split"] == "test"].reset_index(drop=True)
    y_dev, y_test = dev["resolvable"].to_numpy(), test["resolvable"].to_numpy()
    out = {"n_dev": len(dev), "n_test": len(test), "features": cols,
           "resolvable_rate": {"dev": round(float(y_dev.mean()), 4), "test": round(float(y_test.mean()), 4)},
           "answerable_rate": {"dev": round(float(dev["answerable"].mean()), 4),
                               "test": round(float(test["answerable"].mean()), 4)},
           "always_automate": {"coverage": 1.0, "wrong_automation_rate": round(float(1 - y_test.mean()), 4)},
           "models": {}}
    preds = test[["query_id", "split", "answerable", "resolvable"]].copy()
    fitted_models = {}
    for mname, model in candidate_models(with_llm="llm_logit" in feats.columns).items():
        c = columns_for(mname, cols)
        oof = cross_fit(model, dev[c], y_dev)
        fitted = clone(model).fit(dev[c], y_dev)
        p_test = fitted.predict_proba(test[c])[:, 1]
        preds[mname] = p_test
        auto, escalate = thresholds_from(oof, y_dev)
        res = {"label": LABELS[mname], "dev_crossfit": scores(oof, y_dev), "test": scores(p_test, y_test),
               "test_detects_answerable": scores(p_test, test["answerable"].to_numpy()),
               "thresholds": {"auto": auto, "escalate": round(escalate, 4)},
               "test_lanes": apply_thresholds(p_test, y_test, auto["threshold"], escalate),
               "test_curve": risk_coverage_curve(p_test, y_test),
               "other_targets": {}}
        for target in OTHER_TARGETS:
            a, e = thresholds_from(oof, y_dev, target)
            lanes = apply_thresholds(p_test, y_test, a["threshold"], e)
            res["other_targets"][str(target)] = {"dev_coverage": a["coverage"], "test_coverage": lanes["auto"]["share"],
                                                 "test_wrong_automation_rate": lanes["wrong_automation_rate"]}
        if mname == "logistic":
            res["coefficients"] = importance(fitted, c)
        out["models"][mname] = res
        fitted_models[mname] = (fitted, c, auto["threshold"], escalate)

    best = max(("logistic", "boosting"), key=lambda m: out["models"][m]["dev_crossfit"]["average_precision"])
    out["selected_model"] = best

    # Hard rules on top of the selected model
    fitted, c, t_auto, t_esc = fitted_models[best]
    texts = queries.set_index("query_id").loc[test["query_id"], "text"].tolist()
    decisions = [decide(p, t, t_auto, t_esc) for p, t in zip(preds[best], texts)]
    lanes = np.array([d.lane for d in decisions])
    hits = [rule_hits(t) for t in texts]
    rule_counts: dict[str, int] = {}
    for h in hits:
        for rule, _, _ in h:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    auto = lanes == "auto"
    out["with_rules"] = {
        "tickets_matching_a_rule": int(sum(bool(h) for h in hits)), "rule_counts": rule_counts,
        "auto_share": round(float(auto.mean()), 4),
        "wrong_automation_rate": round(float((~y_test[auto]).mean()), 4) if auto.any() else None,
        "lane_shares": {lane: round(float((lanes == lane).mean()), 4) for lane in ("auto", "assist", "escalate")},
    }
    examples = []
    for lane in ("auto", "assist", "escalate"):
        idx = np.flatnonzero(lanes == lane)[:1]
        for i in idx:
            examples.append({"lane": lane, "question": texts[i][:200], "resolvable": bool(y_test[i]),
                             "reasons": decisions[i].reasons})
    out["examples"] = examples
    preds["lane"] = lanes
    return out, preds, {"model": fitted_models[best], "dev": dev, "y_dev": y_dev}


def transfer(source: dict, target_name: str, target_runs: pd.DataFrame, target_queries: pd.DataFrame,
             extra: pd.DataFrame | None = None) -> dict:
    """Apply a model and thresholds trained on one knowledge base, unchanged, to another one's test set."""
    fitted, c, t_auto, t_esc = source["model"]
    feats = build_features(target_runs, target_queries, extra=extra)
    test = feats[feats["split"] == "test"]
    missing = [col for col in c if col not in test.columns]
    if missing:
        return {"skipped": f"missing features {missing}"}
    p = fitted.predict_proba(test[c])[:, 1]
    y = test["resolvable"].to_numpy()
    return {"target": target_name, "test": scores(p, y), "test_lanes": apply_thresholds(p, y, t_auto, t_esc),
            "test_curve": risk_coverage_curve(p, y)}


def plot_risk_coverage(results: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE

    names = [n for n in DATASETS if n in results]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 3.8), squeeze=False, facecolor=SURFACE)
    for ax, name in zip(axes[0], names):
        res = results[name]
        best = res["selected_model"]
        series = [(best, LABELS[best] + " (selected)"), ("score_only", LABELS["score_only"])]
        if "llm_only" in res["models"]:
            series.append(("llm_only", LABELS["llm_only"]))
        for (mname, label), colour in zip(series, SERIES):
            # Skip the first few points: with fewer than 20 tickets the error rate is mostly noise
            curve = [p for p in res["models"][mname]["test_curve"] if p["coverage"] * res["n_test"] >= 20]
            ax.plot([p["coverage"] for p in curve], [p["risk"] for p in curve], color=colour, linewidth=2, label=label,
                    zorder=3)
        lanes = res["models"][best]["test_lanes"]
        if lanes["auto"]["n"]:
            ax.plot([lanes["auto"]["share"]], [lanes["wrong_automation_rate"]], marker="o", markersize=9,
                    color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            ax.annotate(f"chosen threshold: {lanes['auto']['share']:.0%} auto-resolved,\n"
                        f"{lanes['wrong_automation_rate']:.0%} of those wrong",
                        (lanes["auto"]["share"], lanes["wrong_automation_rate"]), textcoords="offset points",
                        xytext=(10, -28), fontsize=8, color=INK_SECONDARY)
        ax.axhline(config.TARGET_RISK, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(1.0, config.TARGET_RISK + 0.015, f"target {config.TARGET_RISK:.0%}", ha="right", fontsize=8,
                color=INK_MUTED)
        always = res["always_automate"]["wrong_automation_rate"]
        ax.plot([1.0], [always], marker="s", markersize=7, color=INK_MUTED, zorder=4)
        ax.text(0.98, always + 0.02, f"automate everything: {always:.0%} wrong", ha="right", fontsize=8, color=INK_MUTED)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Share of tickets auto-resolved (coverage)", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel("Share of auto-resolved tickets that are wrong", fontsize=8.5, color=INK_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        title = {"techqa": "TechQA (IBM Technotes)", "stackexchange": "Stack Exchange"}[name]
        note = "" if lanes["auto"]["n"] else f"\nNo safe threshold ({config.TARGET_RISK:.0%} target, 95% confidence)"
        ax.set_title(f"{title}, test set ({res['n_test']:,} questions){note}", loc="left", fontsize=10, color=INK)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.01, "Lower and further right is better. Curves start at 20 auto-resolved tickets. "
             "Threshold chosen on dev questions only.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run_all(datasets=DATASETS, llm_judge: bool = False) -> dict:
    """llm_judge=True adds the LLM judgement (from `python -m src.answer`) as a signal; outputs get an _llm suffix."""
    suffix = "_llm" if llm_judge else ""
    results, sources, extras = {}, {}, {}
    out_dir = config.DATA_DIR / "decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name in datasets:
        if not (config.DATA_DIR / "retrieval" / f"{name}_runs.parquet").exists():
            print(f"skipping {name}: no retrieval runs")
            continue
        loaded[name] = load(name)
        extras[name] = load_llm_judge(name) if llm_judge else None
        if llm_judge and extras[name] is None:
            raise FileNotFoundError(f"no LLM judgement for {name}; run `python -m src.answer` first")
        results[name], preds, sources[name] = evaluate_dataset(name, *loaded[name], extra=extras[name])
        preds.to_parquet(out_dir / f"{name}_predictions{suffix}.parquet", index=False)
        best = results[name]["selected_model"]
        m = results[name]["models"][best]
        print(f"{name}: selected {best}; test AUROC {m['test']['auroc']}, auto {m['test_lanes']['auto']['share']:.1%} "
              f"of tickets with {m['test_lanes']['wrong_automation_rate']} wrong")
    if "stackexchange" in sources and "techqa" in loaded:
        results["transfer_stackexchange_to_techqa"] = transfer(sources["stackexchange"], "techqa", *loaded["techqa"],
                                                               extra=extras.get("techqa"))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"decision_metrics{suffix}.json").write_text(json.dumps(results, indent=2, default=str))
    if any(n in results for n in DATASETS):
        plot_risk_coverage(results, config.RESULTS_DIR / f"decision_risk_coverage{suffix}.png")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--llm-judge", action="store_true", help="add the LLM judgement from `python -m src.answer`")
    args = parser.parse_args(argv)
    results = run_all(args.datasets, llm_judge=args.llm_judge)
    compact = {k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in results.items()}
    print("DECISION_START")
    print(json.dumps(compact, default=str))
    print("DECISION_END")


if __name__ == "__main__":
    main()
