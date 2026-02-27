"""
Vector Pipeline — Search Module

Performs semantic search against a FAISS index.
"""

import time
import numpy as np
from vector_pipeline.embed import DocumentEmbedder
from vector_pipeline.index import VectorIndexBuilder


class VectorSearcher:
    """Semantic search using FAISS + SentenceTransformers."""

    def __init__(self, embedder: DocumentEmbedder, index_builder: VectorIndexBuilder):
        if index_builder.index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        self.embedder = embedder
        self.index_builder = index_builder

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search for semantically similar pages.

        Returns:
            dict with keys:
                - results: list of {doc, page_num, score}
                - latency_ms: query time in milliseconds (encoding + search)
        """
        start = time.perf_counter()

        # Encode the query
        query_vec = self.embedder.encode_query(query).astype("float32")

        # FAISS search
        scores, indices = self.index_builder.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            page = self.embedder.pages[idx]
            results.append(
                {
                    "doc": page["doc"],
                    "page_num": page["page_num"],
                    "score": float(score),
                    "snippet": page["text"][:200],
                }
            )

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "results": results,
            "latency_ms": latency_ms,
        }

    def batch_search(self, queries: list[str], top_k: int = 5) -> list[dict]:
        """Run search for a list of queries."""
        return [self.search(q, top_k) for q in queries]
