"""ETL data pipeline for processing CSV datasets."""

import csv
import os
from datetime import datetime
from transformers import clean_data, validate_record, enrich_record
from aggregator import aggregate_by_field

class DataPipeline:
    """Extract, Transform, Load pipeline for data processing."""

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.stats = {"total": 0, "valid": 0, "invalid": 0, "enriched": 0}

    def run(self):
        """Execute the full ETL pipeline."""
        raw_data = self.extract()
        cleaned = self.transform(raw_data)
        self.load(cleaned)
        return self.stats

    def extract(self):
        """Extract data from CSV file."""
        records = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
                self.stats["total"] += 1
        return records

    def transform(self, records):
        """Clean, validate, and enrich records."""
        results = []
        for record in records:
            cleaned = clean_data(record)
            if validate_record(cleaned):
                enriched = enrich_record(cleaned)
                results.append(enriched)
                self.stats["valid"] += 1
                self.stats["enriched"] += 1
            else:
                self.stats["invalid"] += 1
        return results

    def load(self, records):
        """Write processed records to output CSV."""
        if not records:
            return
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
