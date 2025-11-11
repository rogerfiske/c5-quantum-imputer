# Tech Stack - C5 Quantum Imputer

**Version**: 1.0
**Last Updated**: 2025-11-11

## Core Technologies

### Python
- **Version**: 3.8+
- **Rationale**: Core language for data science and ML workflows

### Data Science Stack
- **numpy**: 2.1.3 - Numerical computing, array operations
- **pandas**: 2.2.3 - DataFrames, data manipulation, parquet I/O
- **scikit-learn**: 1.5.2 - Machine learning (isotonic regression, logistic regression, model selection)

### Data Persistence
- **pyarrow**: 17.0.0 - Parquet file format (efficient columnar storage)
- **joblib**: 1.4.2 - Model serialization/persistence

### Machine Learning (Existing)
- **lightgbm**: Latest - Gradient boosting ranker (existing baseline model)
- **qiskit**: Latest - Quantum computing framework (existing imputation methods)

### Testing
- **pytest**: Latest - Test framework
- **pytest-cov**: Latest - Code coverage reporting

### Development Tools
- **black**: Code formatting (PEP8)
- **mypy**: Static type checking
- **flake8**: Linting

## Data Format Standards

### Input Data
- **Raw data**: CSV format (`data/raw/c5_Matrix.csv`)
- **Processed data**: Parquet format for efficiency
- **Event structure**: 11,589 lottery events, 5 winners from 39 positions

### Model Artifacts
- **Trained models**: `.pkl` files via joblib
- **Calibrators**: `.joblib` files
- **Configurations**: Python dataclasses or JSON

### Evaluation Outputs
- **Metrics**: JSON format (`.ai/*.json`)
- **Debug logs**: Markdown (`.ai/debug-log.md`)
- **Evaluation reports**: Markdown (`.ai/epic-*-evaluation-results.md`)

## Directory Structure Conventions

```
src/
├── imputation/          # Quantum imputation methods
├── modeling/
│   ├── calibration/     # Calibration modules (NEW: Epic 14)
│   ├── ensembles/       # Ensemble methods (NEW: Epic 15)
│   ├── meta/            # Meta-learning layers (NEW: Epic 14)
│   └── rankers/         # Ranking models (existing LGBM)
├── evaluation/          # Evaluation utilities
└── utils/               # Shared utilities

tests/
├── modeling/
│   ├── calibration/     # Tests for calibration modules
│   ├── ensembles/       # Tests for ensemble modules
│   └── meta/            # Tests for meta-learning
└── integration/         # End-to-end integration tests

production/              # Production models and scripts
data/                    # Dataset storage
docs/                    # Documentation (PRD, architecture, stories)
.ai/                     # Debug logs, evaluation results
```

## Version Compatibility

- All dependencies must be compatible with Python 3.8+
- Pin exact versions in production (`requirements.txt`)
- Use virtual environments for isolation (venv, conda)

## Performance Requirements

- **Calibration**: <5s for 1000 events
- **Full pipeline** (calibration + extremizer + Ising): <10s for 1000 events
- **Memory**: Fit in <4GB RAM for typical datasets

## Future Considerations

- Potential migration to PyTorch/JAX for advanced meta-learning (Epic 16+)
- Integration with MLflow for experiment tracking (Epic 17)
- Dashboard visualization with Plotly (Epic 17)
