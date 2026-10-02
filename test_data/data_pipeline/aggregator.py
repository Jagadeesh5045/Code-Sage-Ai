"""Data aggregation functions."""

from collections import defaultdict

def aggregate_by_field(records, field):
    """Group records by a field and count occurrences."""
    groups = defaultdict(list)
    for record in records:
        key = record.get(field, "unknown")
        groups[key].append(record)
    return {k: len(v) for k, v in groups.items()}

def compute_summary(records):
    """Compute summary statistics for numeric fields."""
    summary = {}
    for key in records[0].keys() if records else []:
        values = []
        for r in records:
            try:
                values.append(float(r[key]))
            except (ValueError, TypeError):
                continue
        if values:
            summary[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    return summary
