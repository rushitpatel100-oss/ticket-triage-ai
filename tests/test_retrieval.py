"""Offline tests for retrieval: no model downloads, a hashing 'embedding' stands in for the real encoder."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from src import config
from src.retrieval.core import Ranking, chunk_documents, doc_starts, reciprocal_rank_fusion, top_k
from src.retrieval.metrics import per_query, summarise
from src.retrieval.retrievers import BM25, DenseRetriever, analyze, rerank


class FakeEncoder:
    """Bag-of-words vectors, L2-normalised: similar words give similar vectors, which is all the tests need."""

    name = "fake-hashing-encoder"

    def __init__(self):
        self.vec = HashingVectorizer(n_features=2048, alternate_sign=False, norm="l2", analyzer=analyze)

    def encode_docs(self, texts):
        return self.vec.transform(list(texts)).toarray().astype(np.float32)

    encode_queries = encode_docs


class OverlapReranker:
    """Scores a pair by the number of shared words."""

    name = "fake-overlap-reranker"

    def score(self, pairs):
        return np.array([len(set(analyze(q)) & set(analyze(p))) for q, p in pairs], dtype=np.float32)


def docs_frame(texts, titles=None):
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(texts))], "title": titles or [""] * len(texts),
                         "text": texts})


def test_analyze_keeps_codes_and_their_parts():
    tokens = analyze("Error 0x80070005 in my-file.txt after upgrade to 4.1.1.1")
    assert {"0x80070005", "my-file.txt", "file", "txt", "4.1.1.1", "upgrad", "error"} <= set(tokens)
    assert "in" not in tokens and "to" not in tokens


def test_top_k_orders_and_pads():
    r = top_k(np.array([[0.1, 0.9, 0.5]]), 5)
    assert r.idx.tolist() == [[1, 2, 0, -1, -1]]
    assert r.scores[0, 0] == pytest.approx(0.9) and np.isneginf(r.scores[0, -1])


def test_bm25_prefers_rare_matching_terms():
    texts = ["printer offline in the office", "vpn disconnects every hour", "printer paper jam",
             "error 0x80070005 when installing updates", "printer driver update"]
    bm25 = BM25().fit(texts)
    r = bm25.search(["windows update fails with 0x80070005", "vpn keeps dropping"], 3)
    assert r.idx[0, 0] == 3 and r.idx[1, 0] == 1
    assert r.scores[0, 0] > r.scores[0, 1]


def test_chunking_overlaps_and_keeps_title():
    words = [f"w{i}" for i in range(250)]
    docs = docs_frame([" ".join(words), "short doc"], titles=["Long title", "Short"])
    chunks = chunk_documents(docs, words=100, overlap=20)
    long = chunks[chunks["doc_pos"] == 0]["text"].tolist()
    assert len(long) == 3  # starts at 0, 80, 160
    assert long[0].startswith("w0 ") and long[1].startswith("Long title\nw80 ")
    assert long[0].split()[-1] == "w99" and "w80" in long[0]  # overlap
    assert chunks[chunks["doc_pos"] == 1]["text"].tolist() == ["short doc"]
    assert doc_starts(chunks["doc_pos"].to_numpy(), 2).tolist() == [0, 3]
    with pytest.raises(ValueError):
        doc_starts(np.array([0, 0, 2]), 3)  # document 1 has no passage


def test_dense_scores_a_document_by_its_best_passage():
    filler = " ".join(f"filler{i}" for i in range(120))
    docs = docs_frame([filler + " vpn certificate expired renew it", "printer offline", "mailbox full"])
    chunks = chunk_documents(docs, words=50, overlap=10)
    starts = doc_starts(chunks["doc_pos"].to_numpy(), len(docs))
    enc = FakeEncoder()
    dense = DenseRetriever(enc, chunks["text"].tolist(), chunks["doc_pos"].to_numpy(), starts)
    r = dense.search(enc.encode_queries(["vpn certificate expired"]), 3)
    assert r.idx[0, 0] == 0  # the answer sits in the document's last passage


def test_reciprocal_rank_fusion_rewards_agreement():
    a = Ranking(np.array([[0, 1, 2]]), np.zeros((1, 3)))
    b = Ranking(np.array([[1, 3, -1]]), np.zeros((1, 3)))
    fused = reciprocal_rank_fusion([a, b], k=60, depth=4)
    assert fused.idx[0].tolist()[0] == 1  # ranked 2nd and 1st beats 1st only
    assert set(fused.idx[0].tolist()) == {0, 1, 2, 3}
    assert fused.scores[0, 0] == pytest.approx(1 / 62 + 1 / 61)


def test_rerank_reorders_candidates():
    docs = docs_frame(["reset your password in the portal", "vpn certificate expired renew it", "printer offline"])
    chunks = chunk_documents(docs)
    starts = doc_starts(chunks["doc_pos"].to_numpy(), len(docs))
    enc = FakeEncoder()
    dense = DenseRetriever(enc, chunks["text"].tolist(), chunks["doc_pos"].to_numpy(), starts)
    queries = ["my vpn certificate expired"]
    candidates = Ranking(np.array([[2, 0, 1]]), np.zeros((1, 3), dtype=np.float32))
    out = rerank(OverlapReranker(), queries, enc.encode_queries(queries), candidates, dense,
                 chunks["text"].tolist(), depth=3, passages_per_doc=2)
    assert out.idx[0, 0] == 1


def test_metrics_by_hand():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "gold"]]
    gold = [["a"], ["missing"], ["gold"]]
    v = per_query(ranked, gold)
    assert v["first_rank"].tolist()[0] == 1 and np.isinf(v["first_rank"][1]) and v["first_rank"][2] == 3
    assert v["rr"].tolist() == [1.0, 0.0, pytest.approx(1 / 3)]
    s = summarise(ranked, gold)
    assert s["hit@1"] == pytest.approx(1 / 3, abs=1e-4) and s["hit@5"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["mrr@10"] == pytest.approx((1 + 1 / 3) / 3, abs=1e-4)
    assert s["ndcg@10"] == pytest.approx((1 + 1 / np.log2(4)) / 3, abs=1e-4)
    lo, hi = s["hit@5_ci95"]
    assert 0 <= lo <= s["hit@5"] <= hi <= 1


def test_run_all_end_to_end(tmp_path, monkeypatch):
    from src.retrieval.__main__ import run_all
    from src.sources import common

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    topics = ["vpn certificate expired", "printer driver crash", "mailbox quota exceeded", "laptop battery drain",
              "excel macro blocked", "wifi authentication loop", "teams camera black", "disk encryption key"]
    docs = pd.DataFrame({"doc_id": [f"kb:{i}" for i in range(len(topics))], "source": "kb", "title": topics,
                         "text": [f"{t}. Fix for {t}: follow the steps." for t in topics], "url": ""})
    rows = []
    for i, t in enumerate(topics * 5):
        answerable = i % 4 != 0
        rows.append({"query_id": f"q{i}", "source": "kb", "text": f"help, {t} again",
                     "answerable": answerable, "gold_doc_ids": [f"kb:{i % len(topics)}"] if answerable else [],
                     "gold_answer": "", "split": ["train", "val", "test"][i % 3]})
    queries = pd.DataFrame(rows)
    common.save_tables("techqa", docs, queries)

    results = run_all(["techqa"], FakeEncoder(), [OverlapReranker()])
    methods = results["techqa"]["methods"]
    assert list(methods) == ["bm25", "dense", "hybrid", "hybrid+rerank:fake-overlap-reranker"]
    assert methods["hybrid+rerank:fake-overlap-reranker"]["label"] == "Hybrid + rerank (fake-overlap-reranker)"
    for m in methods.values():
        assert sum(m["splits"][k]["n_queries"] for k in ("dev", "test")) == int(queries["answerable"].sum())
        assert m["splits"]["test"]["hit@1"] == 1.0  # every question names its topic exactly
        assert set(m["splits"]) == {"train", "val", "dev", "test"}
        assert m["splits"]["dev"]["n_queries"] == m["splits"]["train"]["n_queries"] + m["splits"]["val"]["n_queries"]

    runs = pd.read_parquet(tmp_path / "processed" / "retrieval" / "techqa_runs.parquet")
    assert len(runs) == 4 * len(queries)  # every query, answerable or not, for every method
    assert runs["doc_ids"].map(len).max() <= config.RUN_DEPTH
    saved = json.loads((tmp_path / "results" / "retrieval_metrics.json").read_text())
    assert saved["techqa"]["info"]["rerankers"] == ["fake-overlap-reranker"]
    assert (tmp_path / "results" / "retrieval_comparison.png").exists()

    # A second run reuses the cached passage embeddings and gives the same results
    cached = list((tmp_path / "processed" / "retrieval").glob("techqa_fake-hashing-encoder_*.npy"))
    assert len(cached) == 1
    again = run_all(["techqa"], FakeEncoder(), [])
    assert again["techqa"]["methods"]["dense"]["splits"] == methods["dense"]["splits"]
    assert list(again["techqa"]["methods"]) == ["bm25", "dense", "hybrid"]


def test_paired_comparison(tmp_path, monkeypatch):
    from src.retrieval.compare import compare, paired_bootstrap
    from src.sources import common

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    docs = pd.DataFrame({"doc_id": ["a", "b"], "source": "kb", "title": "", "text": ["x", "y"], "url": ""})
    queries = pd.DataFrame({"query_id": [f"q{i}" for i in range(6)], "source": "kb", "text": "t",
                            "answerable": [True] * 5 + [False], "gold_doc_ids": [["a"]] * 5 + [[]],
                            "gold_answer": "", "split": ["train", "val", "test", "test", "train", "test"]})
    common.save_tables("techqa", docs, queries)
    good = [["a", "b"]] * 6
    bad = [["b", "a"]] * 6  # correct document second: reciprocal rank 0.5
    runs = pd.concat([
        pd.DataFrame({"query_id": queries["query_id"], "split": queries["split"], "answerable": queries["answerable"],
                      "method": m, "doc_ids": d, "scores": [[1.0, 0.5]] * 6})
        for m, d in (("hybrid", bad), ("dense", good))])
    (tmp_path / "processed" / "retrieval").mkdir(parents=True)
    runs.to_parquet(tmp_path / "processed" / "retrieval" / "techqa_runs.parquet", index=False)

    out = compare("techqa", reference="hybrid")["methods"]["dense"]
    assert out["dev"]["n_queries"] == 3 and out["test"]["n_queries"] == 2  # the unanswerable query is skipped
    assert out["test"]["mrr@10"]["mean_diff"] == 0.5
    assert out["test"]["mrr@10"]["share_of_resamples_better"] == 1.0
    assert out["dev"]["hit@5"]["mean_diff"] == 0.0  # both find it within 5

    r = paired_bootstrap(np.array([0.0, 0.0, 1.0, -1.0]))
    assert r["questions_better"] == 1 and r["questions_worse"] == 1 and r["ci95"][0] < 0 < r["ci95"][1]
