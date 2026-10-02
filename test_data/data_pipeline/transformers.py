"""Data transformation functions for the pipeline."""

import re
from datetime import datetime

def clean_data(record):
    """Clean a data record by stripping whitespace and normalizing fields."""
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, str):
            value = value.strip()
            value = re.sub(r"\s+", " ", value)
        cleaned[key.lower().strip()] = value
    return cleaned

def validate_record(record):
    """Validate that a record has required fields and correct format."""
    required = ["name", "email"]
    for field in required:
        if field not in record or not record[field]:
            return False
    if "email" in record and not re.match(r"[^@]+@[^@]+\.[^@]+", record["email"]):
        return False
    return True

def enrich_record(record):
    """Add computed fields to a record."""
    record["processed_at"] = datetime.utcnow().isoformat()
    if "name" in record:
        parts = record["name"].split()
        record["first_name"] = parts[0] if parts else ""
        record["last_name"] = parts[-1] if len(parts) > 1 else ""
    return record
