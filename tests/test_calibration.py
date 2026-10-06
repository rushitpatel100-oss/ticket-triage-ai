"""Offline tests for E1 calibration: synthetic logits with a known temperature, and a fake trainer."""

import json

import numpy as np
import pytest

from src import config
from src.calibration import ece, evaluate_logits, fit_temperature, run_all, softmax, summarise_seeds
from tests.test_pipeline import workspace  # noqa: F401  (pytest fixture)


def overconfident(n=4000, k=5, true_t=3.0, seed=0):
    """Labels drawn from softmax(z); the 'model' reports z * true_t, i.e. it is over-confident by true_t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, k))
    y = np.array([rng.choice(k, p=p) for p in softmax(z)])
    return z * true_t, y


def test_temperature_recovers_known_overconfidence():
    logits, y = overconfident()
    t = fit_temperature(logits[:2000], y[:2000])
    assert t == pytest.approx(3.0, rel=0.15)
    before = ece(softmax(logits[2000:]).max(1), softmax(logits[2000:]).argmax(1) == y[2000:])
    after = ece(softmax(logits[2000:], t).max(1), softmax(logits[2000:], t).argmax(1) == y[2000:])
    assert after < before / 2


def test_scaling_never_changes_predictions():
    logits, y = overconfident(n=1000)
    out = evaluate_logits(logits[:500], y[:500], logits[500:], y[500:])
    assert out["accuracy"] == pytest.approx(float((logits[500:].argmax(1) == y[500:]).mean()), abs=1e-4)
    assert out["after"]["ece"] < out["before"]["ece"]
    assert out["after"]["mean_confidence"] < out["before"]["mean_confidence"]  # over-confidence removed
    assert sum(b["n"] for b in out["before"]["reliability"]) == 500


def test_summary_over_seeds():
    runs = [{"seed": s, "macro_f1": f, "accuracy": f, "temperature": 2.0,
             "before": {"ece": 0.2, "nll": 1.0, "brier": 0.5, "auto_routed_share": 0.9, "auto_routed_accuracy": 0.7},
             "after": {"ece": 0.05, "nll": 0.8, "brier": 0.45, "auto_routed_share": 0.6, "auto_routed_accuracy": None}}
            for s, f in ((1, 0.5), (2, 0.6), (3, 0.7))]
    s = summarise_seeds(runs)
    assert s["macro_f1"]["mean"] == pytest.approx(0.6) and s["macro_f1"]["std"] == pytest.approx(0.1)
    assert s["ece_after"]["mean"] == pytest.approx(0.05) and s["auto_routed_accuracy_after"] is None


def test_run_all_with_fake_trainer(workspace):  # noqa: F811
    calls = []

    def fake_trainer(task, train, val, test, args, seed):
        calls.append((task, seed))
        labels = sorted(train[config.TASKS[task]].unique())
        idx = {lab: i for i, lab in enumerate(labels)}
        y_val, y_test = val[config.TASKS[task]].map(idx).to_numpy(), test[config.TASKS[task]].map(idx).to_numpy()
        rng = np.random.default_rng(seed)

        def logits(y):  # right most of the time, and far too sure of itself
            z = rng.normal(0, 1, (len(y), len(labels)))
            z[np.arange(len(y)), y] += 1.5
            return z * 4
        return {"train_seconds": 1}, {"labels": labels, "val_logits": logits(y_val), "test_logits": logits(y_test),
                                      "y_val": y_val, "y_test": y_test}

    results = run_all(["queue", "type"], seeds=(1, 2), trainer=fake_trainer)
    assert calls == [("queue", 1), ("queue", 2), ("type", 1), ("type", 2)]
    q = results["tasks"]["queue"]
    assert q["distilbert"]["summary"]["seeds"] == [1, 2] and q["distilbert"]["summary"]["temperature"]["mean"] > 1
    assert "tfidf" in q and 0 <= q["tfidf"]["macro_f1"] <= 1
    saved = json.loads((workspace / "results" / "calibration_metrics.json").read_text())
    assert set(saved["tasks"]) == {"queue", "type"}
    assert (workspace / "results" / "calibration_reliability.png").exists()
