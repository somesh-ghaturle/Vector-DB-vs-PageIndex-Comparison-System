import json
import os
import tempfile
import traceback

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from benchmark.forms import BenchmarkForm
from benchmark.models import BenchmarkSession, UploadedPDF, Query

from pageindex_pipeline.index import PageIndexBuilder
from pageindex_pipeline.search import PageIndexSearcher
from vector_pipeline.embed import DocumentEmbedder
from vector_pipeline.index import VectorIndexBuilder
from vector_pipeline.search import VectorSearcher
from evaluation.memory import MemoryTracker
from evaluation.speed import measure_query_latency
from evaluation.accuracy import evaluate_queries


def home(request):
    """Landing page with the upload form."""
    form = BenchmarkForm()
    recent = BenchmarkSession.objects.filter(status="completed")[:5]
    return render(request, "benchmark/home.html", {"form": form, "recent": recent})


def run_benchmark(request):
    """Handle form submission, run both pipelines, save results."""
    if request.method != "POST":
        return redirect("benchmark:home")

    form = BenchmarkForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "benchmark/home.html", {"form": form})

    # Validate PDFs manually (not in the Django form due to multiple file limitation)
    pdf_files = request.FILES.getlist("pdfs")
    if not pdf_files:
        form.add_error(None, "Please upload at least one PDF file.")
        return render(request, "benchmark/home.html", {"form": form})

    # Create session
    session = BenchmarkSession.objects.create(
        top_k=form.cleaned_data["top_k"],
        num_runs=form.cleaned_data["num_runs"],
        status="running",
    )

    # Save uploaded PDFs to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="benchmark_")

    for pdf in pdf_files:
        filepath = os.path.join(tmp_dir, pdf.name)
        with open(filepath, "wb") as f:
            for chunk in pdf.chunks():
                f.write(chunk)
        UploadedPDF.objects.create(
            session=session,
            file=pdf,
            filename=pdf.name,
        )

    # Parse queries
    query_lines = form.cleaned_data["queries"].strip().split("\n")
    relevant_lines = form.cleaned_data.get("relevant_pages", "").strip().split("\n")

    queries_data = []
    for i, line in enumerate(query_lines):
        line = line.strip()
        if not line:
            continue
        relevant = []
        if i < len(relevant_lines) and relevant_lines[i].strip():
            try:
                relevant = json.loads(relevant_lines[i].strip())
            except json.JSONDecodeError:
                relevant = []
        Query.objects.create(session=session, text=line, relevant_pages=relevant)
        queries_data.append({"query": line, "relevant_pages": relevant})

    top_k = session.top_k
    num_runs = session.num_runs

    try:
        mem = MemoryTracker()
        mem.snapshot("baseline")

        # ── PageIndex Pipeline ──────────────────────────────
        pi_builder = PageIndexBuilder()
        pi_builder.add_directory(tmp_dir)
        pi_build_time = pi_builder.build_index()
        mem.snapshot("pageindex_after_build")

        pi_searcher = PageIndexSearcher(pi_builder)

        pi_speed = measure_query_latency(
            pi_searcher.search,
            [q["query"] for q in queries_data],
            top_k=top_k,
            timed_runs=num_runs,
        )
        pi_accuracy = evaluate_queries(pi_searcher.search, queries_data, top_k=top_k)

        session.pi_build_time = round(pi_build_time, 4)
        session.pi_avg_latency = pi_speed["avg_latency_ms"]
        session.pi_p95_latency = pi_speed["p95_latency_ms"]
        session.pi_recall = pi_accuracy["avg_recall@k"]
        session.pi_precision = pi_accuracy["avg_precision@k"]
        session.pi_mrr = pi_accuracy["mrr"]

        mem.snapshot("pageindex_after_eval")

        # ── Vector DB Pipeline ──────────────────────────────
        embedder = DocumentEmbedder()
        embedder.add_directory(tmp_dir)
        mem.snapshot("vector_model_loaded")

        embed_time = embedder.generate_embeddings()
        mem.snapshot("vector_after_embedding")

        v_builder = VectorIndexBuilder(dimension=embedder.embedding_dim)
        v_build_time = v_builder.build_index(embedder.embeddings)
        total_build = embed_time + v_build_time
        mem.snapshot("vector_after_index_build")

        v_searcher = VectorSearcher(embedder, v_builder)

        v_speed = measure_query_latency(
            v_searcher.search,
            [q["query"] for q in queries_data],
            top_k=top_k,
            timed_runs=num_runs,
        )
        v_accuracy = evaluate_queries(v_searcher.search, queries_data, top_k=top_k)

        session.vdb_build_time = round(total_build, 4)
        session.vdb_avg_latency = v_speed["avg_latency_ms"]
        session.vdb_p95_latency = v_speed["p95_latency_ms"]
        session.vdb_recall = v_accuracy["avg_recall@k"]
        session.vdb_precision = v_accuracy["avg_precision@k"]
        session.vdb_mrr = v_accuracy["mrr"]

        # Memory
        baseline_mb = mem.snapshots[0]["absolute_mb"]
        pi_mem = next(s for s in mem.snapshots if s["label"] == "pageindex_after_build")
        v_mem = next(s for s in mem.snapshots if s["label"] == "vector_after_index_build")
        session.pi_memory_mb = round(pi_mem["absolute_mb"] - baseline_mb, 2)
        session.vdb_memory_mb = round(v_mem["absolute_mb"] - baseline_mb, 2)

        # Per-query breakdown
        per_query = []
        for pq, vq in zip(pi_accuracy["per_query"], v_accuracy["per_query"]):
            per_query.append({
                "query": pq["query"],
                "pi_recall": pq["recall@k"],
                "pi_precision": pq["precision@k"],
                "pi_retrieved": pq["retrieved_pages"],
                "vdb_recall": vq["recall@k"],
                "vdb_precision": vq["precision@k"],
                "vdb_retrieved": vq["retrieved_pages"],
                "relevant": pq["relevant_pages"],
            })
        session.per_query_results = per_query
        session.status = "completed"
        session.save()

    except Exception as e:
        session.status = "failed"
        session.error_message = traceback.format_exc()
        session.save()
        return render(request, "benchmark/error.html", {
            "session": session,
            "error": str(e),
        })

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return redirect("benchmark:results", session_id=session.id)


def results(request, session_id):
    """Display benchmark results."""
    session = get_object_or_404(BenchmarkSession, id=session_id)
    return render(request, "benchmark/results.html", {"s": session})


def history(request):
    """List all past benchmark sessions."""
    sessions = BenchmarkSession.objects.all()[:50]
    return render(request, "benchmark/history.html", {"sessions": sessions})
