"""
Evaluation — Speed Benchmarking

Measures index build time and average query latency with warm-up runs.
"""

import time
import statistics


def measure_build_time(build_fn, *args, **kwargs) -> dict:
    """Measure time to build an index.

    Args:
        build_fn: Callable that builds the index.

    Returns:
        dict with build_time_sec.
    """
    start = time.perf_counter()
    build_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return {"build_time_sec": round(elapsed, 4)}


def measure_query_latency(
    search_fn,
    queries: list[str],
    top_k: int = 5,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict:
    """Measure average query latency across multiple runs.

    Args:
        search_fn: Callable(query, top_k) -> dict with latency_ms.
        queries: List of query strings.
        warmup_runs: Runs to discard (cache warming).
        timed_runs: Runs to measure per query.

    Returns:
        dict with avg_latency_ms, median_latency_ms, p95_latency_ms,
        min_latency_ms, max_latency_ms.
    """
    # Warm up
    for q in queries[: min(len(queries), warmup_runs)]:
        search_fn(q, top_k)

    all_latencies = []

    for q in queries:
        query_latencies = []
        for _ in range(timed_runs):
            result = search_fn(q, top_k)
            query_latencies.append(result["latency_ms"])
        all_latencies.extend(query_latencies)

    all_latencies.sort()
    p95_idx = int(len(all_latencies) * 0.95)

    return {
        "avg_latency_ms": round(statistics.mean(all_latencies), 3),
        "median_latency_ms": round(statistics.median(all_latencies), 3),
        "p95_latency_ms": round(all_latencies[p95_idx], 3),
        "min_latency_ms": round(min(all_latencies), 3),
        "max_latency_ms": round(max(all_latencies), 3),
        "total_queries": len(queries),
        "runs_per_query": timed_runs,
    }
