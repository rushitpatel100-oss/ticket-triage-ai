"""BM25 (keywords), dense embeddings (meaning) and a cross-encoder reranker."""

import re
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.retrieval.core import Ranking, top_k

# Words, plus compound tokens such as error codes, versions and file names ("0x80070005", "4.1.1.1",
# "my-file.txt"). Compounds are indexed whole and as their parts, so "file" still matches "my-file.txt".
_TOKEN = re.compile(r"\w[\w.\-]*\w|\w")
_SPLIT = re.compile(r"[.\-]+")
_PORTER = PorterStemmer()


@lru_cache(maxsize=500_000)
def _stem(token: str) -> str:
    """Porter-stem plain words ("updates" -> "updat"); leave codes, versions and file names as they are."""
    return _PORTER.stem(token) if token.isalpha() else token


def analyze(text: str) -> list[str]:
    out = []
    for token in _TOKEN.findall(text.lower()):
        parts = [p for p in _SPLIT.split(token) if p]
        if len(parts) > 1:
            out.extend(parts)
        out.append(token)
    return [_stem(t) for t in out if len(t) > 1 and t not in ENGLISH_STOP_WORDS]


class BM25:
    """Okapi BM25 over a sparse term matrix (k1 and b are the usual defaults)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts) -> "BM25":
        self.vectorizer = CountVectorizer(analyzer=analyze, dtype=np.float32)
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        length = np.asarray(tf.sum(axis=1)).ravel()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = self.k1 * (1 - self.b + self.b * length[rows] / max(length.mean(), 1e-9))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + norm) * idf[tf.indices]
        self.weights = tf.T.tocsr()  # terms x documents
        self.n_docs = n_docs
        return self

    def search(self, queries, k: int, batch_size: int = 256) -> Ranking:
        q = self.vectorizer.transform(queries)
        q.data[:] = 1.0  # each query term counts once
        parts = []
        for start in range(0, q.shape[0], batch_size):
            scores = (q[start:start + batch_size] @ self.weights).toarray()
            parts.append(top_k(scores, k))
        return Ranking(np.vstack([p.idx for p in parts]), np.vstack([p.scores for p in parts]))


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model; embeddings are L2-normalised so dot product = cosine."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 128, max_length: int = 512,
                 fp16: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer  # imported here so tests stay offline

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_length
        self.query_prefix, self.batch_size = query_prefix, batch_size
        self.fp16 = fp16 and torch.cuda.is_available()
        if self.fp16:
            self.model.half()  # about twice as fast on a GPU; similarities change only in the 3rd-4th decimal

    def _encode(self, texts) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=len(texts) > 5000)
        return emb.astype(np.float32)

    def encode_docs(self, texts) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


class DenseRetriever:
    """Embeds every passage; a document's score is the score of its best passage."""

    def __init__(self, encoder, passage_texts, passage_doc_pos: np.ndarray, starts: np.ndarray,
                 passage_emb: np.ndarray | None = None):
        self.encoder = encoder
        self.passage_emb = passage_emb if passage_emb is not None else encoder.encode_docs(passage_texts)
        if len(self.passage_emb) != len(passage_doc_pos):
            raise ValueError("passage embeddings do not match the passages")
        self.passage_doc_pos, self.starts = passage_doc_pos, starts

    def search(self, query_emb: np.ndarray, k: int, batch_size: int = 256) -> Ranking:
        parts = []
        for start in range(0, len(query_emb), batch_size):
            passage_scores = query_emb[start:start + batch_size] @ self.passage_emb.T
            doc_scores = np.maximum.reduceat(passage_scores, self.starts, axis=1)  # best passage per document
            parts.append(top_k(doc_scores, k))
        return Ranking(np.vstack([p.idx for p in parts]), np.vstack([p.scores for p in parts]))

    def best_passages(self, query_vec: np.ndarray, doc: int, n: int) -> np.ndarray:
        """Indexes of a document's n passages most similar to the query."""
        lo = self.starts[doc]
        hi = self.starts[doc + 1] if doc + 1 < len(self.starts) else len(self.passage_doc_pos)
        sims = self.passage_emb[lo:hi] @ query_vec
        return lo + np.argsort(-sims, kind="stable")[:n]


class CrossEncoderReranker:
    """Reads query and passage together (slower but more accurate than comparing two embeddings)."""

    def __init__(self, model_name: str, batch_size: int = 64, max_length: int = 512, fp16: bool = False):
        import torch
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size
        self.fp16 = fp16 and torch.cuda.is_available()
        if self.fp16:
            # Half precision roughly doubles speed on a GPU; rankings barely change.
            module = self.model if isinstance(self.model, torch.nn.Module) else getattr(self.model, "model", None)
            if isinstance(module, torch.nn.Module):
                module.half()
            else:
                self.fp16 = False

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict(pairs, batch_size=self.batch_size,
                                             show_progress_bar=len(pairs) > 20000), dtype=np.float32)


def rerank(reranker, queries, query_emb: np.ndarray, candidates: Ranking, dense: DenseRetriever,
           passage_texts, depth: int, passages_per_doc: int) -> Ranking:
    """Re-score the top `depth` candidates; a document scores as its best passage."""
    pairs, owners = [], []
    for q, text in enumerate(queries):
        for j, doc in enumerate(candidates.idx[q, :depth]):
            if doc < 0:
                continue
            for p in dense.best_passages(query_emb[q], int(doc), passages_per_doc):
                pairs.append((text, passage_texts[p]))
                owners.append((q, j))
    scores = reranker.score(pairs) if pairs else np.array([], dtype=np.float32)

    best = np.full((len(queries), depth), -np.inf, dtype=np.float32)
    for (q, j), s in zip(owners, scores):
        best[q, j] = max(best[q, j], s)
    order = np.argsort(-best, axis=1, kind="stable")
    idx = np.take_along_axis(candidates.idx[:, :depth], order, axis=1)
    sc = np.take_along_axis(best, order, axis=1)
    idx = np.where(np.isfinite(sc), idx, -1)
    return Ranking(idx, sc)
