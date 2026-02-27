"""
PageIndex Pipeline — Search Module

Performs keyword/TF-IDF based search against the PageIndex.
"""

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pageindex_pipeline.index import PageIndexBuilder


class PageIndexSearcher:
    """Search the TF-IDF index built by PageIndexBuilder."""

    def __init__(self, index: PageIndexBuilder):
        if index.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        self.index = index

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search for the query and return top_k results.

        Returns:
            dict with keys:
                - results: list of {doc, page_num, score}
                - latency_ms: query time in milliseconds
        """
        start = time.perf_counter()

        # Transform query using the fitted vectorizer
        query_vec = self.index.vectorizer.transform([query])

        # Cosine similarity against all pages
        scores = cosine_similarity(query_vec, self.index.tfidf_matrix).flatten()

        # Rank by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                page = self.index.pages[idx]
                results.append({
                    "doc": page["doc"],
                    "page_num": page["page_num"],
                    "score": float(scores[idx]),
                    "snippet": page["text"][:200],
                })

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "results": results,
            "latency_ms": latency_ms,
        }

    def batch_search(self, queries: list[str], top_k: int = 5) -> list[dict]:
        """Run search for a list of queries."""
        return [self.search(q, top_k) for q in queries]
