"""E1: can the routing models' confidence be trusted? Temperature scaling and several training seeds.

The demo sends a ticket to a person when the model is less than REVIEW_THRESHOLD (60%) sure. That only
works if "80% sure" really means right about 80% of the time. Modern neural networks are usually
over-confident (Guo et al., 2017). Temperature scaling divides the logits by one number T, fitted on the
validation set, which fixes much of that without changing a single prediction.

    python -m src.calibration                      # DistilBERT x 3 seeds x 3 tasks + TF-IDF (needs a GPU)
    python -m src.calibration --seeds 42 --tasks type --max-train-samples 2000   # quick trial

Writes results/calibration_metrics.json and results/calibration_reliability.png.
"""

import argparse
import json
from argparse import Namespace

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from src import config

SEEDS = (42, 43, 44)
BINS = 15
SPLIT_KEYS = ("before", "after")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The T > 0 that minimises the negative log-likelihood of the true labels (searched on log T)."""
    res = minimize_scalar(lambda log_t: nll(logits, y, float(np.exp(log_t))), bounds=(np.log(0.05), np.log(20.0)),
                          method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"lo": round(float(edges[b]), 4), "hi": round(float(edges[b + 1]), 4), "n": int(m.sum()),
                        "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(correct[m].mean()), 4)})
    return out


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error of the top label: average |accuracy - confidence| over confidence bins."""
    total = len(conf)
    return float(sum(b["n"] / total * abs(b["accuracy"] - b["confidence"]) for b in reliability_bins(conf, correct, bins)))


def probability_report(probs: np.ndarray, y: np.ndarray, threshold: float = config.REVIEW_THRESHOLD) -> dict:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = pred == y
    onehot = np.eye(probs.shape[1])[y]
    auto = conf >= threshold
    return {"ece": round(ece(conf, correct), 4),
            "nll": round(float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()), 4),
            "brier": round(float(((probs - onehot) ** 2).sum(axis=1).mean()), 4),
            "mean_confidence": round(float(conf.mean()), 4),
            # what the demo does: auto-route above the threshold, send the rest to a person
            "auto_routed_share": round(float(auto.mean()), 4),
            "auto_routed_accuracy": round(float(correct[auto].mean()), 4) if auto.any() else None,
            "reliability": reliability_bins(conf, correct)}


def evaluate_logits(val_logits, y_val, test_logits, y_test) -> dict:
    """Fit T on validation, report the test set before and after. Predictions (and so F1) never change."""
    t = fit_temperature(val_logits, y_val)
    pred = np.asarray(test_logits).argmax(axis=1)
    return {"temperature": round(t, 4),
            "accuracy": round(float((pred == y_test).mean()), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro", labels=np.arange(np.shape(test_logits)[1]),
                                             zero_division=0)), 4),
            "before": probability_report(softmax(test_logits), y_test),
            "after": probability_report(softmax(test_logits, t), y_test)}


def summarise_seeds(runs: list[dict]) -> dict:
    """Mean and standard deviation over seeds for the headline numbers."""
    def stat(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1
                else 0.0}
    out = {"seeds": [r["seed"] for r in runs], "macro_f1": stat([r["macro_f1"] for r in runs]),
           "accuracy": stat([r["accuracy"] for r in runs]), "temperature": stat([r["temperature"] for r in runs])}
    for key in SPLIT_KEYS:
        for metric in ("ece", "nll", "brier", "auto_routed_share", "auto_routed_accuracy"):
            out[f"{metric}_{key}"] = stat([r[key][metric] for r in runs])
    return out


def tfidf_logits(task: str, train, val, test) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The TF-IDF baseline as logits (log-probabilities), with C chosen on validation as in src.baseline."""
    import pandas as pd

    from src.baseline import make_tfidf_model
    from src.data import label_names

    col = config.TASKS[task]
    labels = label_names(pd.concat([train, val, test]), task)
    best, best_f1 = None, -1.0
    for c in (0.5, 2.0, 8.0):
        model = make_tfidf_model(c).fit(train["text"], train[col])
        f1 = f1_score(val[col], model.predict(val["text"]), average="macro")
        if f1 > best_f1:
            best, best_f1 = model, f1
    order = [list(best.classes_).index(lab) if lab in best.classes_ else None for lab in labels]

    def logits(df):
        p = best.predict_proba(df["text"])
        full = np.full((len(df), len(labels)), 1e-9)
        for j, k in enumerate(order):
            if k is not None:
                full[:, j] = p[:, k]
        return np.log(full)
    return logits(val), logits(test), labels


def run_all(tasks, seeds=SEEDS, train_args: Namespace | None = None, max_train_samples: int | None = None,
            trainer=None) -> dict:
    """trainer(task, train, val, test, args, seed) -> (metrics, outputs); defaults to src.train.train_task.
    The first seed's model is saved as the demo model; the others only report."""
    from src.data import load_splits

    train, val, test = load_splits()
    if max_train_samples and max_train_samples < len(train):
        train = train.sample(max_train_samples, random_state=config.SEED).reset_index(drop=True)
    if trainer is None:
        from src.train import train_task

        def trainer(task, tr, va, te, args, seed):
            return train_task(task, tr, va, te, args, seed=seed, save=seed == seeds[0])
    args = train_args or Namespace(model_name=config.BASE_MODEL, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                                   lr=config.LEARNING_RATE, max_length=config.MAX_LENGTH, push_to_hub=None,
                                   class_weights=True)
    results = {"review_threshold": config.REVIEW_THRESHOLD, "bins": BINS, "seeds": list(seeds),
               "base_model": args.model_name, "train_rows": len(train), "tasks": {}}
    for task in tasks:
        col = config.TASKS[task]
        if col not in train:
            continue
        runs = []
        for seed in seeds:
            metrics, out = trainer(task, train, val, test, args, seed)
            run = {"seed": seed, **evaluate_logits(out["val_logits"], out["y_val"], out["test_logits"], out["y_test"])}
            run["train_seconds"] = metrics.get("train_seconds")
            runs.append(run)
            print(f"{task} seed {seed}: macro-F1 {run['macro_f1']}, T={run['temperature']}, "
                  f"ECE {run['before']['ece']} -> {run['after']['ece']}")
        tv, tt, labels = tfidf_logits(task, train, val, test)
        y_val = val[col].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        y_test = test[col].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        tfidf = evaluate_logits(tv, y_val, tt, y_test)
        print(f"{task} TF-IDF: macro-F1 {tfidf['macro_f1']}, T={tfidf['temperature']}, "
              f"ECE {tfidf['before']['ece']} -> {tfidf['after']['ece']}")
        results["tasks"][task] = {"labels": out["labels"], "n_val": len(val), "n_test": len(test),
                                  "distilbert": {"summary": summarise_seeds(runs), "runs": runs}, "tfidf": tfidf}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "calibration_metrics.json").write_text(json.dumps(results, indent=2))
    plot_reliability(results, config.RESULTS_DIR / "calibration_reliability.png")
    return results


def plot_reliability(results: dict, path) -> None:
    """Reliability diagrams for DistilBERT (first seed), before and after temperature scaling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE

    tasks = list(results["tasks"])
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.9 * len(tasks), 3.9), squeeze=False, facecolor=SURFACE)
    for ax, task in zip(axes[0], tasks):
        run = results["tasks"][task]["distilbert"]["runs"][0]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        for key, colour, name in (("before", SERIES[1], "Before"), ("after", SERIES[0], f"After (T = {run['temperature']:.2f})")):
            pts = [b for b in run[key]["reliability"] if b["n"] >= 10]
            ax.plot([b["confidence"] for b in pts], [b["accuracy"] for b in pts], marker="o", markersize=4,
                    color=colour, linewidth=2, label=f"{name}: ECE {run[key]['ece']:.3f}", zorder=3)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{task} (seed {run['seed']})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Model confidence", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel("Share actually correct", fontsize=8.5, color=INK_SECONDARY)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.text(0.01, 0.01, "On the dashed line, confidence matches accuracy. Bins with fewer than 10 test tickets are "
             "hidden. Temperature fitted on the validation set.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=list(config.TASKS), default=list(config.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args(argv)
    results = run_all(args.tasks, tuple(args.seeds), max_train_samples=args.max_train_samples)
    compact = {t: {"distilbert": v["distilbert"]["summary"],
                   "tfidf_ece": (v["tfidf"]["before"]["ece"], v["tfidf"]["after"]["ece"])}
               for t, v in results["tasks"].items()}
    print(json.dumps(compact))


if __name__ == "__main__":
    main()
