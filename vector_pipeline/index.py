"""
Vector Pipeline — FAISS Index Module

Builds and manages a FAISS vector index for semantic search.
"""

import time
import numpy as np
import faiss


class VectorIndexBuilder:
    """Wraps FAISS index construction with timing and metadata."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.build_time: float = 0.0

    def build_index(self, embeddings: np.ndarray, use_ivf: bool = False) -> float:
        """Build a FAISS index from embeddings.

        Args:
            embeddings: (N, D) numpy array of normalized vectors.
            use_ivf: If True, use IVFFlat for larger datasets.
                     If False, use IndexFlatIP (exact, brute-force).

        Returns:
            Build time in seconds.
        """
        n_vectors = embeddings.shape[0]
        start = time.perf_counter()

        if use_ivf and n_vectors > 1000:
            n_clusters = min(int(np.sqrt(n_vectors)), 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings.astype("float32"))
            self.index.add(embeddings.astype("float32"))
            self.index.nprobe = min(10, n_clusters)
        else:
            # Exact inner-product search (fast for < 100k vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings.astype("float32"))

        self.build_time = time.perf_counter() - start
        return self.build_time

    def save(self, path: str = "vector.index") -> None:
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str = "vector.index") -> None:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def __repr__(self) -> str:
        return f"VectorIndexBuilder(dim={self.dimension}, vectors={self.num_vectors})"
