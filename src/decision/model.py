"""Resolvability models, risk-controlled thresholds and evaluation."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def candidate_models(seed: int = config.SEED) -> dict:
    """A one-signal baseline, an interpretable model and a non-linear one."""
    return {
        "score_only": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "logistic": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
        "boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                   l2_regularization=1.0, min_samples_leaf=20, random_state=seed),
    }


def columns_for(name: str, all_columns: list[str], primary: str = config.PRIMARY_RETRIEVER) -> list[str]:
    return [f"{primary}_top1"] if name == "score_only" else all_columns


def cross_fit(model, X: pd.DataFrame, y: np.ndarray, folds: int = config.CV_FOLDS,
              seed: int = config.SEED) -> np.ndarray:
    """Out-of-fold probabilities: every question is scored by a model that never saw it."""
    out = np.zeros(len(y))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        m = clone(model).fit(X.iloc[train_idx], y[train_idx])
        out[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    return out


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted probability and observed rate."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - y[mask].mean())
    return float(total)


def scores(probs: np.ndarray, y: np.ndarray) -> dict:
    both = len(np.unique(y)) > 1
    return {"auroc": round(float(roc_auc_score(y, probs)), 4) if both else None,
            "average_precision": round(float(average_precision_score(y, probs)), 4) if both else None,
            "brier": round(float(brier_score_loss(y, probs)), 4), "ece": round(ece(probs, y), 4),
            "positive_rate": round(float(y.mean()), 4), "n": int(len(y))}


def clopper_pearson_upper(errors: int, n: int, confidence: float = config.RISK_CONFIDENCE) -> float:
    """One-sided upper confidence bound on an error rate."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, n - errors))


def risk_controlled_threshold(probs: np.ndarray, ok: np.ndarray, target: float = config.TARGET_RISK,
                              confidence: float = config.RISK_CONFIDENCE) -> dict:
    """Lowest threshold whose wrong-automation rate is <= target with the given confidence.

    Selection with guaranteed risk (SGR, Geifman & El-Yaniv 2017): binary search over the sorted scores,
    testing about log2(n) thresholds, each with a Clopper-Pearson bound at a Bonferroni-corrected level, so
    the chosen threshold's true error rate is below the reported bound with the requested confidence
    (assuming new tickets look like the calibration tickets).
    """
    order = np.argsort(-probs, kind="stable")
    p_sorted, wrong_sorted = probs[order], ~ok[order].astype(bool)
    n = len(probs)
    none = {"threshold": float("inf"), "coverage": 0.0, "observed_risk": None, "risk_upper_bound": None,
            "n_auto": 0, "thresholds_tested": 0}
    if n == 0:
        return none
    steps = max(int(np.ceil(np.log2(n))), 1)
    level = 1 - (1 - confidence) / steps  # Bonferroni over the thresholds the search tests
    cum_wrong = np.cumsum(wrong_sorted)
    best, lo, hi = none, 0, n - 1
    for _ in range(steps):
        mid = (lo + hi + 1) // 2
        theta = p_sorted[mid]
        n_auto = int(np.searchsorted(-p_sorted, -theta, side="right"))  # everything scoring >= theta
        errors = int(cum_wrong[n_auto - 1])
        upper = clopper_pearson_upper(errors, n_auto, level)
        if upper <= target:
            if n_auto > best["n_auto"]:
                best = {"threshold": float(theta), "coverage": round(n_auto / n, 4),
                        "observed_risk": round(errors / n_auto, 4), "risk_upper_bound": round(upper, 4),
                        "n_auto": n_auto}
            lo = mid
        else:
            hi = mid - 1
        if lo >= hi:
            break
    best["thresholds_tested"] = steps
    return best


def recall_threshold(probs: np.ndarray, ok: np.ndarray, recall: float = config.ASSIST_RECALL) -> float:
    """Highest threshold that still keeps `recall` of the resolvable tickets at or above it."""
    pos = np.sort(probs[ok.astype(bool)])[::-1]
    if len(pos) == 0:
        return 1.0
    k = int(np.ceil(recall * len(pos))) - 1
    return float(pos[min(max(k, 0), len(pos) - 1)])


def apply_thresholds(probs: np.ndarray, ok: np.ndarray, t_auto: float, t_escalate: float) -> dict:
    """Share of tickets per lane and how many in each lane were really resolvable."""
    ok = ok.astype(bool)
    lanes = np.where(probs >= t_auto, "auto", np.where(probs >= t_escalate, "assist", "escalate"))
    out = {}
    for lane in ("auto", "assist", "escalate"):
        mask = lanes == lane
        out[lane] = {"share": round(float(mask.mean()), 4), "n": int(mask.sum()),
                     "resolvable_rate": round(float(ok[mask].mean()), 4) if mask.any() else None}
    auto = lanes == "auto"
    out["wrong_automation_rate"] = round(float((~ok[auto]).mean()), 4) if auto.any() else None
    out["resolvable_missed_by_escalation"] = round(float((lanes[ok] == "escalate").mean()), 4) if ok.any() else None
    return out


def risk_coverage_curve(probs: np.ndarray, ok: np.ndarray, points: int = 50) -> list[dict]:
    order = np.argsort(-probs, kind="stable")
    wrong = np.cumsum(~ok[order].astype(bool))
    n = len(probs)
    idx = np.unique(np.linspace(1, n, min(points, n)).astype(int))
    return [{"coverage": round(i / n, 4), "risk": round(float(wrong[i - 1] / i), 4)} for i in idx]


def importance(model, columns: list[str]) -> list[tuple[str, float]]:
    """Standardised coefficients for the logistic model (sign = direction)."""
    lr = model[-1] if hasattr(model, "__getitem__") else None
    if not isinstance(lr, LogisticRegression):
        return []
    coefs = lr.coef_.ravel()
    order = np.argsort(-np.abs(coefs))
    return [(columns[i], round(float(coefs[i]), 3)) for i in order]
