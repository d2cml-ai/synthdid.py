# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of Synthetic Difference-in-Differences (SDID), a causal inference method for policy evaluation and treatment effect estimation in panel data. The library combines synthetic control and difference-in-differences approaches.

## Development Commands

### Package Management
- **Install dependencies**: `pipenv install` (uses Pipfile for dependency management)
- **Install dev dependencies**: `pipenv install --dev`
- **Activate environment**: `pipenv shell`
- **Install package in development mode**: `pip install -e .`

### Running Code
- **Run examples**: `jupyter notebook examples/examples.ipynb`
- **Import main class**: `from synthdid.synthdid import Synthdid`
- **Load sample data**: `from synthdid.get_data import california_prop99`

### Testing
⚠️ **No test suite exists** - this is a critical gap that should be addressed. Consider adding:
- `pytest` for testing framework
- `tests/` directory with unit tests for each module
- Test coverage for core estimation methods

## Code Architecture

### Core Design Pattern
The library uses a **method chaining pattern** where users:
1. Initialize `Synthdid` with data and column specifications
2. Call `.fit()` to estimate the model
3. Call `.vcov()` to compute standard errors
4. Call `.summary()` to format results

### Key Modules

**`synthdid/synthdid.py`**: Main entry point that inherits from multiple mixins:
- Combines functionality from SDID, Variance, Plots, and Summary classes
- Provides unified interface for all estimation methods

**`synthdid/sdid.py`**: Core estimation logic
- `SDID` class with `fit()` method
- Supports three modes: 'sdid', 'sc' (synthetic control), 'did' (difference-in-differences)
- Handles staggered adoption designs

**`synthdid/solver.py`**: Optimization algorithms
- Frank-Wolfe method implementation for weight estimation
- Covariate adjustment with "optimized" and "projected" methods

**`synthdid/utils.py`**: Data preprocessing
- Panel data matrix construction
- Data validation and transformation utilities

**`synthdid/vcov.py`**: Variance estimation
- Multiple standard error methods: placebo, bootstrap, jackknife
- `Variance` class with flexible inference options

**`synthdid/plots.py`**: Visualization capabilities
- Outcome trajectory plots
- Weight visualization for units and time periods

### Data Flow
1. Raw panel data → `utils.py` preprocessing → structured matrices
2. Matrices → `solver.py` optimization → estimated weights
3. Weights + data → `sdid.py` estimation → treatment effects
4. Results → `vcov.py` inference → standard errors
5. Final results → `summary.py` + `plots.py` → formatted output

## Common Usage Patterns

### Basic Estimation
```python
from synthdid.synthdid import Synthdid
from synthdid.get_data import california_prop99

df = california_prop99()
result = Synthdid(df, "State", "Year", "treated", "PacksPerCapita").fit().vcov().summary()
```

### Staggered Adoption
When treatment occurs at different times across units, the library automatically handles staggered designs.

### Covariate Adjustment
Use `covariates` parameter with method "optimized" or "projected" for incorporating additional control variables.

## Data Requirements

Input data must be a pandas DataFrame with:
- **Unit column**: Identifier for panel units (e.g., states, countries)
- **Time column**: Time period identifier
- **Treatment column**: Binary treatment indicator
- **Outcome column**: Dependent variable
- Optional covariate columns

## Dependencies

Core dependencies (pinned versions in requirements.txt):
- numpy (1.23.5)
- pandas (1.5.3) 
- matplotlib (3.7.1)
- scipy (1.10.1)
- statsmodels (0.13.5)

Development environment targets Python 3.8+ with Pipenv for dependency management.

## Known Issues and Limitations

- **No test suite**: Critical gap in code quality assurance
- **Empty `__init__.py`**: Main classes not exposed at package level
- **Inconsistent error handling**: Mix of print statements and exceptions
- **Missing type hints**: Reduces code maintainability
- **Rigid dependency pinning**: May cause compatibility issues

## Development Priorities

1. Add comprehensive test suite with pytest
2. Implement proper package-level imports in `__init__.py`
3. Add type hints throughout codebase
4. Standardize error handling with custom exceptions
5. Add code formatting and linting tools