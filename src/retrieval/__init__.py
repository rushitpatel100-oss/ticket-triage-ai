"""Retrieval over the knowledge base: BM25, dense embeddings, hybrid fusion and a cross-encoder reranker.

All retrievers return a `Ranking`: two arrays of shape (n_queries, k) holding document positions
(-1 where fewer than k documents exist) and their scores, best first.
"""
