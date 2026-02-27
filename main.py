"""
main.py — Orchestrator for Vector DB vs PageIndex Comparison

Runs both pipelines on the same documents, evaluates them head-to-head,
and produces a results CSV + terminal summary.

Usage:
    python main.py                          # default: data/docs/
    python main.py --docs path/to/pdfs/     # custom document directory
    python main.py --top-k 10 --runs 30     # tweak evaluation params
"""

import argparse
import json
import os
import sys

import pandas as pd

from evaluation.memory import MemoryTracker
from evaluation.speed import measure_query_latency
from evaluation.accuracy import evaluate_queries

# ── Helpers ─────────────────────────────────────────────────────────


def load_queries(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metric(label: str, value, unit: str = "") -> None:
    print(f"  {label:<30} {value:>10} {unit}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Vector DB vs PageIndex Comparison Benchmark"
    )
    parser.add_argument(
        "--docs", default="data/docs", help="Directory containing PDF files"
    )
    parser.add_argument(
        "--queries", default="data/queries.json", help="Path to queries JSON"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="Timed runs per query for latency"
    )
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.docs):
        print(f"Error: Document directory '{args.docs}' not found.")
        print("Please add PDF files to the data/docs/ directory.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(args.docs) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Error: No PDF files found in '{args.docs}'.")
        print("Please add at least one PDF to the data/docs/ directory.")
        sys.exit(1)

    queries = load_queries(args.queries)
    print(f"\n📄 Found {len(pdf_files)} PDF(s) in {args.docs}")
    print(f"❓ Loaded {len(queries)} evaluation queries")

    mem = MemoryTracker()
    mem.snapshot("baseline")

    results = {}

    # ── PIPELINE 1: PageIndex (TF-IDF) ─────────────────────────────

    print_header("PIPELINE 1: PageIndex (TF-IDF)")

    from pageindex_pipeline.index import PageIndexBuilder
    from pageindex_pipeline.search import PageIndexSearcher

    pi_builder = PageIndexBuilder()
    n_pages = pi_builder.add_directory(args.docs)
    print(f"  Ingested {n_pages} pages from {len(pdf_files)} PDF(s)")

    # Build index
    pi_build_time = pi_builder.build_index()
    print_metric("Index build time", f"{pi_build_time:.4f}", "sec")
    mem.snapshot("pageindex_after_build")

    # Search
    pi_searcher = PageIndexSearcher(pi_builder)

    # Speed
    pi_speed = measure_query_latency(
        pi_searcher.search,
        [q["query"] for q in queries],
        top_k=args.top_k,
        timed_runs=args.runs,
    )
    print_metric("Avg query latency", f"{pi_speed['avg_latency_ms']:.3f}", "ms")
    print_metric("P95 query latency", f"{pi_speed['p95_latency_ms']:.3f}", "ms")

    # Accuracy
    pi_accuracy = evaluate_queries(pi_searcher.search, queries, top_k=args.top_k)
    print_metric(f"Recall@{args.top_k}", f"{pi_accuracy['avg_recall@k']:.4f}", "")
    print_metric(f"Precision@{args.top_k}", f"{pi_accuracy['avg_precision@k']:.4f}", "")
    print_metric("MRR", f"{pi_accuracy['mrr']:.4f}", "")

    mem.snapshot("pageindex_after_eval")

    results["PageIndex"] = {
        "build_time_sec": round(pi_build_time, 4),
        "avg_latency_ms": pi_speed["avg_latency_ms"],
        "p95_latency_ms": pi_speed["p95_latency_ms"],
        f"recall@{args.top_k}": pi_accuracy["avg_recall@k"],
        f"precision@{args.top_k}": pi_accuracy["avg_precision@k"],
        "mrr": pi_accuracy["mrr"],
    }

    # ── PIPELINE 2: Vector DB (FAISS) ──────────────────────────────

    print_header("PIPELINE 2: Vector DB (FAISS + SentenceTransformers)")

    from vector_pipeline.embed import DocumentEmbedder
    from vector_pipeline.index import VectorIndexBuilder
    from vector_pipeline.search import VectorSearcher

    embedder = DocumentEmbedder()
    n_pages_v = embedder.add_directory(args.docs)
    print(f"  Ingested {n_pages_v} pages from {len(pdf_files)} PDF(s)")
    mem.snapshot("vector_model_loaded")

    # Embed
    embed_time = embedder.generate_embeddings()
    print_metric("Embedding time", f"{embed_time:.4f}", "sec")
    mem.snapshot("vector_after_embedding")

    # Build FAISS index
    v_builder = VectorIndexBuilder(dimension=embedder.embedding_dim)
    v_build_time = v_builder.build_index(embedder.embeddings)
    total_build = embed_time + v_build_time
    print_metric("FAISS index build time", f"{v_build_time:.4f}", "sec")
    print_metric("Total build time", f"{total_build:.4f}", "sec")
    mem.snapshot("vector_after_index_build")

    # Search
    v_searcher = VectorSearcher(embedder, v_builder)

    # Speed
    v_speed = measure_query_latency(
        v_searcher.search,
        [q["query"] for q in queries],
        top_k=args.top_k,
        timed_runs=args.runs,
    )
    print_metric("Avg query latency", f"{v_speed['avg_latency_ms']:.3f}", "ms")
    print_metric("P95 query latency", f"{v_speed['p95_latency_ms']:.3f}", "ms")

    # Accuracy
    v_accuracy = evaluate_queries(v_searcher.search, queries, top_k=args.top_k)
    print_metric(f"Recall@{args.top_k}", f"{v_accuracy['avg_recall@k']:.4f}", "")
    print_metric(f"Precision@{args.top_k}", f"{v_accuracy['avg_precision@k']:.4f}", "")
    print_metric("MRR", f"{v_accuracy['mrr']:.4f}", "")

    mem.snapshot("vector_after_eval")

    results["VectorDB"] = {
        "build_time_sec": round(total_build, 4),
        "avg_latency_ms": v_speed["avg_latency_ms"],
        "p95_latency_ms": v_speed["p95_latency_ms"],
        f"recall@{args.top_k}": v_accuracy["avg_recall@k"],
        f"precision@{args.top_k}": v_accuracy["avg_precision@k"],
        "mrr": v_accuracy["mrr"],
    }

    # ── Memory Summary ─────────────────────────────────────────────

    print_header("MEMORY USAGE")
    for snap in mem.get_report():
        print_metric(snap["label"], f"{snap['absolute_mb']:.1f}", "MB")

    pi_mem = next(s for s in mem.snapshots if s["label"] == "pageindex_after_build")
    v_mem = next(s for s in mem.snapshots if s["label"] == "vector_after_index_build")
    baseline = mem.snapshots[0]["absolute_mb"]

    results["PageIndex"]["memory_mb"] = round(pi_mem["absolute_mb"] - baseline, 2)
    results["VectorDB"]["memory_mb"] = round(v_mem["absolute_mb"] - baseline, 2)

    # ── Results CSV ────────────────────────────────────────────────

    df = pd.DataFrame(results).T
    df.index.name = "Method"
    df.to_csv(args.output)
    print(f"\n✅ Results saved to {args.output}")

    # ── Side-by-Side Summary ───────────────────────────────────────

    print_header("FINAL COMPARISON")
    print(df.to_string())
    print()

    # ── Per-Query Accuracy Breakdown ───────────────────────────────

    print_header("PER-QUERY ACCURACY BREAKDOWN")
    print(f"  {'Query':<35} {'PI Recall':>10} {'VDB Recall':>11}")
    print("  " + "-" * 58)
    for pq, vq in zip(pi_accuracy["per_query"], v_accuracy["per_query"]):
        q = pq["query"][:33]
        print(f"  {q:<35} {pq['recall@k']:>10.4f} {vq['recall@k']:>11.4f}")

    # ── Key Insight ────────────────────────────────────────────────

    print_header("KEY INSIGHT")
    pi_faster = (
        results["PageIndex"]["avg_latency_ms"] < results["VectorDB"]["avg_latency_ms"]
    )
    vdb_better_recall = (
        results["VectorDB"][f"recall@{args.top_k}"]
        > results["PageIndex"][f"recall@{args.top_k}"]
    )

    if pi_faster and vdb_better_recall:
        print("  PageIndex is FASTER but Vector DB has BETTER RECALL.")
        print("  → Use PageIndex for latency-critical keyword search.")
        print("  → Use Vector DB when semantic understanding matters.")
    elif pi_faster:
        print("  PageIndex wins on BOTH speed and accuracy!")
        print("  → Vector DB overhead not justified for this dataset.")
    else:
        print("  Vector DB outperforms on both speed and recall.")
        print("  → Consider dataset size and query complexity.")

    print()


if __name__ == "__main__":
    main()
