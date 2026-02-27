"""
Evaluation — Accuracy Metrics

Computes Recall@K, Precision@K, and MRR against ground-truth annotations.
"""

from __future__ import annotations


def recall_at_k(
    retrieved_pages: list[int], relevant_pages: list[int], k: int = 5
) -> float:
    """Fraction of relevant pages found in top-K results.

    recall@k = |retrieved ∩ relevant| / |relevant|
    """
    if not relevant_pages:
        return 0.0
    retrieved_set = set(retrieved_pages[:k])
    relevant_set = set(relevant_pages)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(
    retrieved_pages: list[int], relevant_pages: list[int], k: int = 5
) -> float:
    """Fraction of top-K results that are relevant.

    precision@k = |retrieved ∩ relevant| / k
    """
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_pages[:k])
    relevant_set = set(relevant_pages)
    return len(retrieved_set & relevant_set) / k


def reciprocal_rank(retrieved_pages: list[int], relevant_pages: list[int]) -> float:
    """Reciprocal rank of the first relevant result.

    MRR = 1 / rank_of_first_relevant
    """
    relevant_set = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_queries(
    search_fn,
    queries: list[dict],
    top_k: int = 5,
) -> dict:
    """Run accuracy evaluation across all queries.

    Args:
        search_fn: Callable(query_text, top_k) -> dict with "results" list.
        queries: List of dicts with "query", "relevant_pages" keys.
        top_k: Number of results to retrieve.

    Returns:
        dict with avg_recall, avg_precision, mrr, per_query details.
    """
    per_query = []
    total_recall = 0.0
    total_precision = 0.0
    total_rr = 0.0

    for q in queries:
        query_text = q["query"]
        relevant = q["relevant_pages"]

        result = search_fn(query_text, top_k)
        retrieved = [r["page_num"] for r in result["results"]]

        r = recall_at_k(retrieved, relevant, top_k)
        p = precision_at_k(retrieved, relevant, top_k)
        rr = reciprocal_rank(retrieved, relevant)

        total_recall += r
        total_precision += p
        total_rr += rr

        per_query.append(
            {
                "query": query_text,
                "relevant_pages": relevant,
                "retrieved_pages": retrieved,
                "recall@k": round(r, 4),
                "precision@k": round(p, 4),
                "reciprocal_rank": round(rr, 4),
            }
        )

    n = len(queries) if queries else 1

    return {
        "avg_recall@k": round(total_recall / n, 4),
        "avg_precision@k": round(total_precision / n, 4),
        "mrr": round(total_rr / n, 4),
        "top_k": top_k,
        "num_queries": len(queries),
        "per_query": per_query,
    }
