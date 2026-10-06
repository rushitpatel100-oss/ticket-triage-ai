"""Offline tests for the resolve / assist / escalate decision."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.decision.features import build_features, feature_columns
from src.decision.model import (
    apply_thresholds,
    clopper_pearson_upper,
    ece,
    recall_threshold,
    risk_controlled_threshold,
)
from src.decision.policy import decide, rule_hits


def synthetic(n=900, seed=0):
    """Questions whose correct article is found with a clear margin (resolvable) or not at all."""
    rng = np.random.default_rng(seed)
    queries, runs = [], []
    for i in range(n):
        qid = f"q{i}"
        resolvable = rng.random() < 0.5
        answerable = resolvable or rng.random() < 0.3
        gold = f"d{i}"
        split = ["train", "val", "test"][i % 3]
        queries.append({"query_id": qid, "source": "s", "text": "printer jam " * int(rng.integers(1, 30)),
                        "answerable": answerable, "gold_doc_ids": [gold] if answerable else [],
                        "gold_answer": "", "split": split})
        others = [f"x{i}_{k}" for k in range(20)]
        if resolvable:
            docs = [gold] + others[:19]
            top = 0.75 + 0.15 * rng.random()
            dense_scores = [top] + list(np.sort(rng.uniform(0.4, 0.6, 19))[::-1])
        else:
            docs = others
            dense_scores = list(np.sort(rng.uniform(0.45, 0.65, 20))[::-1])
        bm25_docs = docs if resolvable and rng.random() < 0.8 else list(reversed(docs))
        for method, d, s in (("dense", docs, dense_scores),
                             ("bm25", bm25_docs, list(np.sort(rng.uniform(5, 15, 20))[::-1])),
                             ("hybrid", docs, list(np.linspace(0.03, 0.01, 20)))):
            runs.append({"query_id": qid, "split": split, "answerable": answerable, "method": method,
                         "doc_ids": d, "scores": s})
    return pd.DataFrame(runs), pd.DataFrame(queries)


def test_features_and_labels():
    runs, queries = synthetic(60)
    feats = build_features(runs, queries)
    assert len(feats) == 60
    r = feats.set_index("query_id")
    resolvable = queries.set_index("query_id").apply(
        lambda q: bool(q["answerable"]) and r.loc[q.name, "dense_gap12"] > 0.09, axis=1)
    assert (r["resolvable"] == resolvable.reindex(r.index)).mean() > 0.9
    cols = feature_columns(feats)
    assert {"dense_top1", "dense_gap12", "dense_z1", "bm25_top1", "agree_top5_overlap_bm25",
            "agree_bm25_rank_of_dense_top1", "q_log_words"} <= set(cols)
    assert "resolvable" not in cols and "answerable" not in cols
    with pytest.raises(ValueError):
        build_features(runs[runs["method"] != "dense"], queries)

    # An unanswerable question stays unresolvable even if its record lists the retrieved document
    q2 = queries.copy()
    first = q2.index[0]
    q2.at[first, "answerable"] = False
    q2.at[first, "gold_doc_ids"] = [runs[(runs["query_id"] == q2.at[first, "query_id"]) & (runs["method"] == "dense")]
                                    ["doc_ids"].iloc[0][0]]
    assert not build_features(runs, q2).set_index("query_id").at[q2.at[first, "query_id"], "resolvable"]


def test_clopper_pearson_rule_of_three():
    # With no errors, about 29 tickets are needed before a 10% error rate can be claimed with 95% confidence
    assert clopper_pearson_upper(0, 29) < 0.10 < clopper_pearson_upper(0, 28)
    assert clopper_pearson_upper(5, 5) == 1.0 and clopper_pearson_upper(0, 0) == 1.0


def test_risk_controlled_threshold():
    probs = np.linspace(1, 0, 200)
    ok = np.r_[np.ones(100, bool), np.zeros(100, bool)]  # perfect ranking
    t = risk_controlled_threshold(probs, ok, target=0.10)
    assert t["observed_risk"] <= t["risk_upper_bound"] <= 0.10
    assert 0.4 <= t["coverage"] <= 0.55  # close to the 100 good ones, a little short because of the bound

    # One wrong ticket at the very top does not block automation (binary search, not a fixed start)
    ok2 = ok.copy()
    ok2[0] = False
    assert risk_controlled_threshold(probs, ok2, target=0.10)["coverage"] > 0.3

    # A stricter confidence level can only reduce coverage
    assert risk_controlled_threshold(probs, ok, 0.10, confidence=0.99)["coverage"] <= t["coverage"]

    rng = np.random.default_rng(1)
    noise = risk_controlled_threshold(rng.random(300), rng.random(300) < 0.5, target=0.10)
    assert noise["coverage"] == 0.0 and noise["threshold"] == float("inf")


def test_guarantee_holds_across_calibration_samples():
    """The point of the method: the chosen threshold's true risk exceeds the target in at most ~5% of samples."""
    rng = np.random.default_rng(2)
    violations, certified = 0, 0
    for _ in range(200):
        p = rng.random(2000)
        ok = rng.random(2000) < p                  # well-calibrated scores: P(ok) = p
        t = risk_controlled_threshold(p, ok, target=0.10)
        if t["coverage"] > 0:
            certified += 1
            true_risk = (1 - t["threshold"]) / 2   # E[1 - p | p >= theta] for p ~ U(0, 1)
            violations += true_risk > 0.10
    assert certified > 100  # a usable threshold is found in most samples
    assert violations / 200 <= 0.05


def test_recall_threshold_and_lanes():
    probs = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ok = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0], bool)
    t_esc = recall_threshold(probs, ok, recall=0.8)  # keep 4 of the 5 resolvable
    assert t_esc == 0.6
    lanes = apply_thresholds(probs, ok, t_auto=0.85, t_escalate=t_esc)
    assert lanes["auto"]["n"] == 2 and lanes["assist"]["n"] == 3 and lanes["escalate"]["n"] == 5
    assert lanes["wrong_automation_rate"] == 0.0
    assert lanes["resolvable_missed_by_escalation"] == 0.2


def test_ece():
    y = np.array([1, 0] * 50)
    assert ece(np.full(100, 0.5), y) == pytest.approx(0.0)
    assert ece(np.full(100, 0.9), y) == pytest.approx(0.4)


def test_rules_only_make_decisions_more_cautious():
    assert decide(0.99, "How do I reset my VPN token?", 0.8, 0.3).lane == "auto"
    d = decide(0.99, "I clicked a phishing link and entered my password", 0.8, 0.3)
    assert d.lane == "escalate" and any("security incident" in r for r in d.reasons)
    assert decide(0.99, "Please grant me access to the finance share", 0.8, 0.3).lane == "assist"
    assert decide(0.99, "Printer offline", 0.8, 0.3, priority="High").lane == "assist"
    assert decide(0.99, "Upgrade the database cluster", 0.8, 0.3, itil_type="Change").lane == "assist"
    assert decide(0.1, "Printer offline", 0.8, 0.3).lane == "escalate"   # rules never relax a decision
    assert rule_hits("antivirus update failed") == []                    # whole words only
    assert decide(0.5, "Printer offline", 0.8, 0.3).reasons[0].startswith("confidence 0.500; below the auto-resolve")
    no_auto = decide(0.99, "Printer offline", float("inf"), 0.3)
    assert no_auto.lane == "assist" and "no auto-resolve threshold met the error target" in no_auto.reasons[0]


def test_run_all_end_to_end(tmp_path, monkeypatch):
    from src.decision.__main__ import run_all
    from src.sources import common

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "processed" / "retrieval").mkdir(parents=True)
    docs = pd.DataFrame({"doc_id": ["d"], "source": "s", "title": "", "text": "t", "url": ""})
    for name, seed in (("techqa", 0), ("stackexchange", 1)):
        runs, queries = synthetic(900, seed)
        common.save_tables(name, docs, queries.assign(gold_doc_ids=[[] for _ in range(len(queries))],
                                                      answerable=False))
        # save_tables checks gold documents exist; write the real labels directly afterwards
        queries.to_parquet(tmp_path / "processed" / name / "queries.parquet", index=False)
        runs.to_parquet(tmp_path / "processed" / "retrieval" / f"{name}_runs.parquet", index=False)

    results = run_all()
    tq = results["techqa"]
    assert tq["n_dev"] == 600 and tq["n_test"] == 300
    best = tq["models"][tq["selected_model"]]
    assert best["test"]["auroc"] > 0.95 > tq["models"]["score_only"]["test"]["auroc"] - 0.2
    lanes = best["test_lanes"]
    assert lanes["auto"]["share"] > 0.2
    assert lanes["wrong_automation_rate"] <= 0.12   # target 10%, small test-set noise allowed
    assert best["thresholds"]["escalate"] <= best["thresholds"]["auto"]["threshold"]
    assert set(tq["with_rules"]["lane_shares"]) == {"auto", "assist", "escalate"}
    assert {e["lane"] for e in tq["examples"]} <= {"auto", "assist", "escalate"}
    assert tq["models"]["logistic"]["coefficients"]
    assert results["transfer_stackexchange_to_techqa"]["test"]["auroc"] > 0.9

    saved = json.loads((tmp_path / "results" / "decision_metrics.json").read_text())
    assert set(saved) == {"techqa", "stackexchange", "transfer_stackexchange_to_techqa"}
    assert (tmp_path / "results" / "decision_risk_coverage.png").exists()
    preds = pd.read_parquet(tmp_path / "processed" / "decision" / "techqa_predictions.parquet")
    assert len(preds) == 300 and {"score_only", "logistic", "boosting", "lane"} <= set(preds.columns)
