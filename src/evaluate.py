"""Shared evaluation helpers: metrics, confusion-matrix plots, error analysis, results log."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config

# Chart colours (light surface so the PNGs read well on GitHub in light and dark mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order: one colour per model
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "n_test": int(len(y_true)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)


def plot_confusion_matrix(y_true, y_pred, labels: list[str], title: str, path: Path) -> None:
    """Row-normalised confusion matrix: each row shows where tickets of that true class ended up."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)

    size = max(4.5, 0.75 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1, size), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)  # empty cells blend into the background
    ax.imshow(np.ma.masked_where(cm == 0, share), cmap=cmap, vmin=0, vmax=1)
    # thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] == 0:
                continue
            colour = "#ffffff" if share[i, j] > 0.5 else INK
            ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=8, color=colour)

    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    _style_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_errors(texts, y_true, y_pred, confidence, path: Path, n: int = 50) -> None:
    """Save the model's most confident mistakes; reading these is the best way to find label noise."""
    df = pd.DataFrame({"text": texts, "actual": y_true, "predicted": y_pred, "confidence": confidence})
    errors = df[df["actual"] != df["predicted"]].sort_values("confidence", ascending=False).head(n)
    errors["confidence"] = errors["confidence"].round(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(path, index=False)


def log_result(task: str, model_name: str, metrics: dict) -> None:
    """Add one model's scores to results/metrics.json (kept in git, used for the README table)."""
    metrics_file = config.RESULTS_DIR / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    results.setdefault(task, {})[model_name] = metrics
    metrics_file.write_text(json.dumps(results, indent=2))


def plot_model_comparison(results: dict, path: Path) -> None:
    """Grouped bar chart of macro-F1 for every model on every task."""
    tasks = list(results)
    models = list(dict.fromkeys(m for task in tasks for m in results[task]))
    bar_h = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 1.1 * len(tasks)), facecolor=SURFACE)
    for k, model in enumerate(models):
        ys = [i + (k - (len(models) - 1) / 2) * bar_h for i in range(len(tasks))]
        vals = [results[t].get(model, {}).get("macro_f1", 0) for t in tasks]
        ax.barh(ys, vals, height=bar_h - 0.04, color=SERIES[k % len(SERIES)], label=model)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(range(len(tasks)), tasks)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro F1 on the test set (higher is better)", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Model comparison by task", color=INK, fontsize=11, loc="left", pad=30)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(models),
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
