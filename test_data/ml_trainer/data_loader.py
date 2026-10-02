"""Dataset loading and preprocessing."""

import csv
import random

def load_dataset(filepath):
    """Load a CSV dataset and split into features and labels."""
    X, y = [], []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            X.append([float(v) for v in row[:-1]])
            y.append(row[-1])
    return X, y

def split_data(X, y, test_size=0.2, random_seed=42):
    """Split data into train and test sets."""
    random.seed(random_seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - test_size))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test
