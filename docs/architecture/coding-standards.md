# Coding Standards - C5 Quantum Imputer

**Version**: 1.0
**Last Updated**: 2025-11-11

## Python Style Guide

### PEP 8 Compliance
- Follow [PEP 8](https://pep8.org/) style guide
- Use `black` for automated formatting (line length: 100)
- Use `flake8` for linting

### Type Hints
- **REQUIRED** for all function signatures
- Use `from __future__ import annotations` for forward references
- Example:
```python
def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ClassName":
    """Fit the model."""
    pass
```

### Docstrings
- **REQUIRED** for all public classes, methods, and functions
- Use **Google-style** or **NumPy-style** docstrings
- Include: Description, Args, Returns, Raises (if applicable)
- Example:
```python
def recall_at_20(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Recall@20 = average hits / 5, expressed as percentage.

    Args:
        y_true: Binary labels (n_events, 39).
        p: Probabilities (n_events, 39).

    Returns:
        Recall@20 as percentage.
    """
    pass
```

## Code Organization

### Module Structure
- One class per file (unless tightly coupled helper classes)
- Group related functions at module level
- Order: imports → constants → helper functions → classes → CLI/main

### Imports
- Standard library first
- Third-party packages second
- Local imports third
- Alphabetize within groups
- Example:
```python
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.utils.metrics import compute_recall
```

### Naming Conventions
- **Classes**: PascalCase (`PositionIsotonicCalibrator`)
- **Functions/Methods**: snake_case (`fit_transform`, `recall_at_20`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- **Private methods**: Leading underscore (`_compute_diagnostics`)

## Data Handling

### NumPy Arrays
- Always specify dtype explicitly: `scores.to_numpy(dtype=float)`
- Document expected shapes in docstrings: `(n_events, 39)`
- Use vectorized operations (avoid Python loops where possible)

### Pandas DataFrames
- Use `.to_numpy()` for conversion (not `.values`)
- Specify column lists explicitly: `score_cols = [f"p{i}" for i in range(1, 40)]`
- Use parquet format for I/O efficiency

### Data Validation
- Validate input shapes at entry points
- Use assertions for internal consistency checks
- Example:
```python
assert scores.shape[1] == 39, f"Expected 39 positions, got {scores.shape[1]}"
assert self._fitted, "Calibrator not fitted. Call fit() first."
```

## Error Handling

### Assertions vs Exceptions
- **Assertions**: Internal consistency, development-time checks
- **Exceptions**: User input validation, runtime errors
- Provide informative error messages

### Numerical Stability
- Always clip probabilities: `np.clip(p, 1e-12, 1.0)`
- Check for division by zero: `np.where(denominator <= 0, default_value, numerator / denominator)`
- Use epsilon for log operations: `np.log(x + eps)`

## Configuration Management

### Dataclasses for Configs
- Use `@dataclass` for configuration objects
- Provide sensible defaults
- Example:
```python
@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    normalize: bool = True
    epsilon: float = 1e-12
```

## CLI Design

### Argument Parsing
- Use `argparse` for CLI interfaces
- Provide `--help` descriptions
- Use subcommands for complex tools (`fit`, `apply`, etc.)
- Example:
```python
parser = argparse.ArgumentParser(description="Position-wise isotonic calibration")
parser.add_argument("--scores", required=True, help="Path to scores.parquet")
```

## Performance Considerations

### Vectorization
- Prefer NumPy vectorized operations over loops
- Profile code for bottlenecks (`cProfile`, `line_profiler`)
- Target: <5s for 1000 events (calibration), <10s full pipeline

### Memory Efficiency
- Use generators for large datasets
- Release large objects explicitly if needed: `del large_array`
- Avoid unnecessary copies: use views where safe

## Comments

### When to Comment
- **Complex algorithms**: Explain the "why," not the "what"
- **Non-obvious choices**: Document rationale (e.g., "Use clip_j=2.0 to avoid noisy couplings")
- **TODOs**: Use `# TODO(name): description` format

### When NOT to Comment
- Obvious code (e.g., `# Increment counter` for `i += 1`)
- Code that can be clarified with better naming

## File Headers

All new modules should include a header:
```python
"""
Module Name - Brief Description
================================
Path: src/path/to/module.py

Longer description of purpose, key classes, and usage.

Expected data formats, outputs, dependencies.
"""
from __future__ import annotations
```

## Version Control

### Commit Messages
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Reference stories/epics: `feat(epic-14): implement position isotonic calibration`
- Keep commits atomic (one logical change per commit)

### Branch Strategy
- `main`: Production-ready code
- `epic-N`: Long-lived epic branches (e.g., `epic-14`)
- `story-N.M`: Short-lived story branches (e.g., `story-14.1`)

## Code Review Checklist

Before marking a story complete:
- [ ] All functions have type hints
- [ ] All public APIs have docstrings
- [ ] Code passes `flake8` linting
- [ ] Code is formatted with `black`
- [ ] Unit tests achieve ≥90% coverage
- [ ] Integration tests pass
- [ ] No hardcoded paths or magic numbers
- [ ] Error messages are informative
