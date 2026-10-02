"""Machine learning model training pipeline."""

import json
from data_loader import load_dataset, split_data
from model import create_model, evaluate_model

class ModelTrainer:
    """End-to-end ML training pipeline with evaluation."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = {}

    def run_pipeline(self, data_path):
        """Execute the full training pipeline."""
        # Step 1: Load data
        X, y = load_dataset(data_path)
        print(f"Loaded {len(X)} samples")

        # Step 2: Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=self.config.get("test_size", 0.2)
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Step 3: Train model
        model_type = self.config.get("model_type", "random_forest")
        self.model = create_model(model_type, self.config.get("params", {}))
        self.model.fit(X_train, y_train)
        print(f"Model trained: {model_type}")

        # Step 4: Evaluate
        self.metrics = evaluate_model(self.model, X_test, y_test)
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        return self.metrics

    def save_results(self, output_path):
        """Save training results to JSON."""
        with open(output_path, "w") as f:
            json.dump({"config": self.config, "metrics": self.metrics}, f, indent=2)
