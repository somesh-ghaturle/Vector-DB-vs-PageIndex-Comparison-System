"""
Vector Pipeline — Embedding Module

Generates sentence embeddings for PDF page text using SentenceTransformers.
"""

import os
import time
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Extracts PDF text and generates dense embeddings per page."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast & good quality

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.pages: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.embed_time: float = 0.0

    # ── Document Ingestion ──────────────────────────────────────────

    def add_pdf(self, filepath: str) -> int:
        """Extract text from every page of a PDF.

        Returns the number of pages added.
        """
        doc = fitz.open(filepath)
        doc_name = os.path.basename(filepath)
        count = 0
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            if text:
                self.pages.append(
                    {
                        "doc": doc_name,
                        "page_num": page_num + 1,
                        "text": text,
                    }
                )
                count += 1
        doc.close()
        return count

    def add_directory(self, dirpath: str) -> int:
        """Add all PDFs in a directory. Returns total pages added."""
        total = 0
        for fname in sorted(os.listdir(dirpath)):
            if fname.lower().endswith(".pdf"):
                total += self.add_pdf(os.path.join(dirpath, fname))
        return total

    # ── Embedding Generation ────────────────────────────────────────

    def generate_embeddings(self, batch_size: int = 32) -> float:
        """Encode all page texts into dense vectors.

        Returns time taken in seconds.
        """
        if not self.pages:
            raise ValueError("No pages to embed. Add documents first.")

        texts = [p["text"] for p in self.pages]

        start = time.perf_counter()
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embed_time = time.perf_counter() - start
        return self.embed_time

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into a dense vector."""
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ── Utilities ───────────────────────────────────────────────────

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def __repr__(self) -> str:
        embedded = self.embeddings is not None
        return (
            f"DocumentEmbedder(model={self.model_name!r}, "
            f"pages={self.num_pages}, embedded={embedded})"
        )
