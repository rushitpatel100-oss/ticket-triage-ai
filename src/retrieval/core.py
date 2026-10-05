"""Building blocks: passages, the Ranking type, top-k selection and reciprocal rank fusion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class Ranking:
    """Document positions (n_queries, k), -1 = empty slot, and their scores, best first."""

    idx: np.ndarray
    scores: np.ndarray

    def top(self, k: int) -> "Ranking":
        return Ranking(self.idx[:, :k], self.scores[:, :k])


def top_k(scores: np.ndarray, k: int) -> Ranking:
    """Best k columns per row of a dense score matrix."""
    n_q, n_docs = scores.shape
    k_eff = min(k, n_docs)
    part = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    sc = np.take_along_axis(part_scores, order, axis=1)
    if k_eff < k:
        idx = np.pad(idx, ((0, 0), (0, k - k_eff)), constant_values=-1)
        sc = np.pad(sc, ((0, 0), (0, k - k_eff)), constant_values=-np.inf)
    return Ranking(idx.astype(np.int64), sc.astype(np.float32))


def chunk_documents(docs: pd.DataFrame, words: int = config.CHUNK_WORDS,
                    overlap: int = config.CHUNK_OVERLAP) -> pd.DataFrame:
    """Split each document into overlapping passages of `words` words (at least one per document).

    Passages after the first start with the document title, so they keep their context.
    Returns columns doc_pos (row position in `docs`) and text, ordered by doc_pos.
    """
    if not 0 <= overlap < words:
        raise ValueError("overlap must be smaller than words")
    step = words - overlap
    doc_pos, texts = [], []
    for pos, (title, text) in enumerate(zip(docs["title"].fillna(""), docs["text"])):
        tokens = str(text).split()
        starts = range(0, max(len(tokens) - overlap, 1), step)
        for start in starts:
            piece = " ".join(tokens[start:start + words])
            if start > 0 and title:
                piece = f"{title}\n{piece}"
            doc_pos.append(pos)
            texts.append(piece)
    return pd.DataFrame({"doc_pos": np.array(doc_pos, dtype=np.int64), "text": texts})


def doc_starts(chunk_doc_pos: np.ndarray, n_docs: int) -> np.ndarray:
    """Index of each document's first passage. Requires passages ordered by document, one or more each."""
    starts = np.searchsorted(chunk_doc_pos, np.arange(n_docs))
    if len(chunk_doc_pos) and (np.bincount(chunk_doc_pos, minlength=n_docs) == 0).any():
        raise ValueError("every document needs at least one passage")
    return starts


def reciprocal_rank_fusion(rankings: list[Ranking], k: int = config.RRF_K,
                           depth: int = config.CANDIDATES) -> Ranking:
    """Combine rankings by summing 1 / (k + rank). Uses ranks only, so scores on different scales mix safely."""
    n_q = rankings[0].idx.shape[0]
    idx = np.full((n_q, depth), -1, dtype=np.int64)
    scores = np.full((n_q, depth), -np.inf, dtype=np.float32)
    for q in range(n_q):
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking.idx[q]):
                if doc >= 0:
                    fused[int(doc)] = fused.get(int(doc), 0.0) + 1.0 / (k + rank + 1)
        best = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:depth]
        for j, (doc, score) in enumerate(best):
            idx[q, j], scores[q, j] = doc, score
    return Ranking(idx, scores)
