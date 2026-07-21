"""Offline metrics, computed purely from the JSONL so they're reproducible and
decoupled from the run that produced it.
"""

import json

from .results import ErrorClass

def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = q / 100 * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)

def analyze(path: str) -> dict:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"error": "no rows"}

    total = len(rows)
    by_class: dict[str, int] = {}
    for row in rows:
        by_class[row["error_class"]] = by_class.get(row["error_class"], 0) + 1

    useful = by_class.get(ErrorClass.OK.value, 0)
    wall_seconds = max(r["end_unix"] for r in rows) - min(r["start_unix"] for r in rows)
    wall_seconds = max(wall_seconds, 1e-9)

    ok_latencies = sorted(r["total_latency_s"] for r in rows if r["error_class"] == ErrorClass.OK.value)
    total_cost = sum(r["cost_usd"] for r in rows)

    return {
        "requests": total,
        "useful": useful,
        "wall_seconds": round(wall_seconds, 2),
        "attempted_rps": round(total / wall_seconds, 1),
        "goodput_rps": round(useful / wall_seconds, 1),
        "goodput_rpm": round(useful / wall_seconds * 60, 0),
        "error_rates": {cls: round(n / total, 4) for cls, n in sorted(by_class.items())},
        "latency_s": {
            "p50": round(_percentile(ok_latencies, 50), 3),
            "p95": round(_percentile(ok_latencies, 95), 3),
            "p99": round(_percentile(ok_latencies, 99), 3),
        },
        "total_cost_usd": round(total_cost, 4),
        "cost_per_1k_useful_usd": round(total_cost / useful * 1000, 4) if useful else None,
    }

def save_analysis(jsonl_path: str) -> dict:
    """Compute `analyze(jsonl_path)` and write it alongside as `<path>.analysis.json`."""
    summary = analyze(jsonl_path)
    out_path = jsonl_path.removesuffix(".jsonl") + ".analysis.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
