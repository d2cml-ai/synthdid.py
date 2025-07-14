# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-06-18

### Added
- Modern development setup using `uv` for dependency management
- Support for Python 3.13
- Comprehensive installation documentation in README

### Changed
- **BREAKING**: Minimum Python version updated from 3.8 to 3.13
- **NumPy**: Updated from 1.23.5 to 2.3.0 (major version upgrade)
- **Pandas**: Updated from 1.5.3 to 2.3.0 (major version upgrade)
- **Matplotlib**: Updated from 3.7.1 to 3.10.3
- **SciPy**: Updated from 1.10.1 to 1.15.3
- **Statsmodels**: Updated from 0.13.5 to 0.14.4
- Git repository moved from `d2cml-ai/synthdid.py` to `workhelix/synthdid.py`
- Simplified dependency management in `setup.py` - removed unnecessary development dependencies
- Updated `Pipfile` to use Python 3.13 and latest package versions
- Streamlined `requirements.txt` to include only core scientific packages

### Technical Details
- All dependencies now use minimum version constraints (`>=`) instead of exact pinning for better compatibility
- Virtual environment creation updated to use `uv venv --python 3.13`
- Removed obsolete dependencies like `black==19.3b0`, `click==7.0`, and other development tools from install_requires

### Testing
- Verified compatibility with updated dependencies
- Core functionality tested with California Prop 99 example dataset
- All package imports working correctly with new versions

### Migration Notes
- Existing installations will need to upgrade to Python 3.13+
- Some deprecation warnings may appear with Pandas 2.3.0 due to updated groupby behavior
- NumPy 2.0+ includes breaking changes from NumPy 1.x - see [NumPy 2.0 migration guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

## [0.10.1] - Previous Release
- Previous stable version with Python 3.8 support and older dependencies