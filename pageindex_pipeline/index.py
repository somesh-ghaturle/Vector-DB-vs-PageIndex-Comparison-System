"""
PageIndex Pipeline — Indexing Module

Builds a keyword/TF-IDF based index from PDF documents.
No embeddings required — relies on lexical matching.
"""

import os
import time
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle


class PageIndexBuilder:
    """Indexes PDF pages using TF-IDF for fast keyword-based retrieval."""

    def __init__(self):
        self.pages: list[dict] = []          # {doc, page_num, text}
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.build_time: float = 0.0

    # ── Document Ingestion ──────────────────────────────────────────

    def add_pdf(self, filepath: str) -> int:
        """Extract text from every page of a PDF and store it.

        Returns the number of pages added.
        """
        doc = fitz.open(filepath)
        doc_name = os.path.basename(filepath)
        count = 0
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            if text:
                self.pages.append({
                    "doc": doc_name,
                    "page_num": page_num + 1,  # 1-indexed
                    "text": text,
                })
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

    # ── Index Building ──────────────────────────────────────────────

    def build_index(self) -> float:
        """Build the TF-IDF index. Returns build time in seconds."""
        if not self.pages:
            raise ValueError("No pages to index. Add documents first.")

        start = time.perf_counter()

        corpus = [p["text"] for p in self.pages]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.build_time = time.perf_counter() - start
        return self.build_time

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str = "pageindex.pkl") -> None:
        """Serialize the index to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "pages": self.pages,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)

    def load(self, path: str = "pageindex.pkl") -> None:
        """Load a previously saved index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pages = data["pages"]
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]

    # ── Utilities ───────────────────────────────────────────────────

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def __repr__(self) -> str:
        indexed = self.tfidf_matrix is not None
        return f"PageIndexBuilder(pages={self.num_pages}, indexed={indexed})"
