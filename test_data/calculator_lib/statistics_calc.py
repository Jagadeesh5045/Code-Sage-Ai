"""Statistical calculator for data analysis."""

import math
from typing import List

def mean(data: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(data) / len(data)

def median(data: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    if not data:
        raise ValueError("Cannot calculate median of empty list")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]

def variance(data: List[float], population: bool = True) -> float:
    """Calculate variance. Use population=False for sample variance."""
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")
    m = mean(data)
    ss = sum((x - m) ** 2 for x in data)
    divisor = len(data) if population else len(data) - 1
    return ss / divisor

def std_deviation(data: List[float], population: bool = True) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(data, population))

def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile (0-100) of the data."""
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
