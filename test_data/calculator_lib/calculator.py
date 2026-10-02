"""Advanced calculator library with mathematical operations."""

import math
from typing import List, Union

Number = Union[int, float]

class Calculator:
    """A full-featured calculator supporting basic and scientific operations."""

    def __init__(self):
        self.history: List[str] = []
        self.memory: float = 0.0

    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers and return the result."""
        result = a + b
        self._record(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a and return the result."""
        result = a - b
        self._record(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: Number, b: Number) -> Number:
        """Multiply two numbers and return the result."""
        result = a * b
        self._record(f"{a} * {b} = {result}")
        return result

    def divide(self, a: Number, b: Number) -> float:
        """Divide a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self._record(f"{a} / {b} = {result}")
        return result

    def power(self, base: Number, exponent: Number) -> Number:
        """Raise base to the power of exponent."""
        result = base ** exponent
        self._record(f"{base} ^ {exponent} = {result}")
        return result

    def sqrt(self, n: Number) -> float:
        """Calculate the square root of n. Raises ValueError if n < 0."""
        if n < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(n)
        self._record(f"sqrt({n}) = {result}")
        return result

    def factorial(self, n: int) -> int:
        """Calculate n factorial. Raises ValueError if n < 0."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        result = math.factorial(n)
        self._record(f"{n}! = {result}")
        return result

    def memory_store(self, value: Number):
        """Store a value in calculator memory."""
        self.memory = float(value)

    def memory_recall(self) -> float:
        """Recall the value stored in calculator memory."""
        return self.memory

    def memory_clear(self):
        """Clear calculator memory."""
        self.memory = 0.0

    def get_history(self) -> List[str]:
        """Return list of all calculations performed."""
        return self.history.copy()

    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()

    def _record(self, entry: str):
        """Record a calculation in history."""
        self.history.append(entry)
