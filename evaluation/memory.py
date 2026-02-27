"""
Evaluation — Memory Profiling

Measures RSS memory usage at key points using psutil.
"""

import os
import psutil


def get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class MemoryTracker:
    """Track memory usage across pipeline stages."""

    def __init__(self):
        self.snapshots: list[dict] = []
        self._baseline = get_memory_mb()

    def snapshot(self, label: str) -> dict:
        """Take a memory snapshot with a descriptive label.

        Returns:
            dict with label, absolute_mb, delta_mb (from baseline).
        """
        current = get_memory_mb()
        entry = {
            "label": label,
            "absolute_mb": round(current, 2),
            "delta_mb": round(current - self._baseline, 2),
        }
        self.snapshots.append(entry)
        return entry

    def get_report(self) -> list[dict]:
        """Return all snapshots as a list of dicts."""
        return self.snapshots.copy()

    def summary(self) -> dict:
        """Return peak memory and total delta."""
        if not self.snapshots:
            return {}
        peak = max(s["absolute_mb"] for s in self.snapshots)
        return {
            "baseline_mb": round(self._baseline, 2),
            "peak_mb": round(peak, 2),
            "peak_delta_mb": round(peak - self._baseline, 2),
            "snapshots": len(self.snapshots),
        }
