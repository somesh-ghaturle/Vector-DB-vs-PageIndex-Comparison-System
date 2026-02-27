import uuid
from django.db import models


class BenchmarkSession(models.Model):
    """A single benchmark run comparing PageIndex vs Vector DB."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("running", "Running"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )
    top_k = models.IntegerField(default=5)
    num_runs = models.IntegerField(default=20)

    # PageIndex results
    pi_build_time = models.FloatField(null=True, blank=True)
    pi_avg_latency = models.FloatField(null=True, blank=True)
    pi_p95_latency = models.FloatField(null=True, blank=True)
    pi_recall = models.FloatField(null=True, blank=True)
    pi_precision = models.FloatField(null=True, blank=True)
    pi_mrr = models.FloatField(null=True, blank=True)
    pi_memory_mb = models.FloatField(null=True, blank=True)

    # VectorDB results
    vdb_build_time = models.FloatField(null=True, blank=True)
    vdb_avg_latency = models.FloatField(null=True, blank=True)
    vdb_p95_latency = models.FloatField(null=True, blank=True)
    vdb_recall = models.FloatField(null=True, blank=True)
    vdb_precision = models.FloatField(null=True, blank=True)
    vdb_mrr = models.FloatField(null=True, blank=True)
    vdb_memory_mb = models.FloatField(null=True, blank=True)

    # Per-query JSON results
    per_query_results = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Benchmark {self.id.hex[:8]} — {self.status}"


class UploadedPDF(models.Model):
    """A PDF uploaded for a benchmark session."""

    session = models.ForeignKey(
        BenchmarkSession, on_delete=models.CASCADE, related_name="pdfs"
    )
    file = models.FileField(upload_to="pdfs/")
    filename = models.CharField(max_length=255)
    num_pages = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


class Query(models.Model):
    """A search query with optional ground-truth relevant pages."""

    session = models.ForeignKey(
        BenchmarkSession, on_delete=models.CASCADE, related_name="queries"
    )
    text = models.CharField(max_length=500)
    relevant_pages = models.JSONField(
        default=list, blank=True, help_text="List of relevant page numbers"
    )

    class Meta:
        verbose_name_plural = "queries"

    def __str__(self):
        return self.text[:60]
