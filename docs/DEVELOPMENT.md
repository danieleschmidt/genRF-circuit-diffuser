# Development Guide

This guide covers development setup, workflows, and best practices for GenRF Circuit Diffuser.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- NgSpice (for SPICE simulation)
- Docker (optional, for containerized development)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/genRF-circuit-diffuser.git
cd genRF-circuit-diffuser

# Install development dependencies
make install-dev

# Verify installation
make test
```

### Development Dependencies

The project uses several development tools:

- **Code Formatting**: Black, isort
- **Linting**: flake8, mypy
- **Testing**: pytest, pytest-cov
- **Pre-commit Hooks**: pre-commit
- **Documentation**: Sphinx

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
make check  # Runs formatting, linting, and tests

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Code Quality Checks

Before committing, always run:

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Or run everything
make check
```

### 3. Pre-commit Hooks

Pre-commit hooks are automatically installed with `make install-dev`. They run:

- Trailing whitespace removal
- YAML syntax checking
- Python code formatting (Black, isort)
- Linting (flake8, mypy)
- Large file detection

## Testing Strategy

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests with SPICE
├── performance/    # Performance benchmarks
└── fixtures/       # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run performance benchmarks
python benchmarks/run_benchmarks.py
```

### Writing Tests

Follow these guidelines:

- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (SPICE, file system)
- Include integration tests for SPICE simulation
- Add performance tests for critical paths

Example test:

```python
def test_circuit_generation_with_valid_spec():
    \"\"\"Test circuit generation with valid specification.\"\"\"
    spec = DesignSpec(
        circuit_type="LNA",
        frequency=2.4e9,
        gain_min=15
    )
    
    diffuser = CircuitDiffuser()
    circuit = diffuser.generate(spec)
    
    assert circuit is not None
    assert circuit.gain >= 15
```

## Code Style Guidelines

### Python Style

- Follow PEP 8 with Black formatting (88 character line length)
- Use type hints for all public APIs
- Write docstrings in Google style
- Import order: standard library, third-party, local

### Documentation

- Document all public classes and methods
- Include code examples in docstrings
- Update README.md for user-facing changes
- Add tutorials for new features

### Commit Messages

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Performance Considerations

### Benchmarking

Run benchmarks regularly:

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Run specific scenario
python benchmarks/run_benchmarks.py --scenario LNA_2G4

# Compare with baseline
python benchmarks/run_benchmarks.py --compare-baseline
```

### Profiling

For performance analysis:

```bash
# Profile circuit generation
python -m cProfile -o profile.stats your_script.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

### Memory Usage

Monitor memory usage during development:

```bash
# Use memory profiler
pip install memory-profiler
python -m memory_profiler your_script.py

# Or use pytest-benchmark
pytest tests/performance/ --benchmark-only
```

## Docker Development

### Development Container

```bash
# Build development image
docker-compose build genrf-dev

# Run development server
docker-compose up genrf-dev

# Run tests in container
docker-compose run genrf-dev make test
```

### Jupyter Development

```bash
# Start Jupyter Lab
docker-compose up genrf-jupyter

# Access at http://localhost:8888
```

## Debugging

### Common Issues

1. **SPICE simulation failures**
   - Check NgSpice installation: `ngspice -v`
   - Verify netlist syntax
   - Check model file paths

2. **Import errors**
   - Ensure package is installed in development mode: `pip install -e .`
   - Check PYTHONPATH

3. **Test failures**
   - Run with verbose output: `pytest -v`
   - Check fixture availability
   - Verify test dependencies

### Debug Tools

- Use `pdb` for interactive debugging
- Add logging with appropriate levels
- Use IDE debuggers (VS Code, PyCharm)

## CI/CD Integration

The project uses GitHub Actions for CI/CD:

- **CI Pipeline**: Runs tests, linting, security checks
- **Security Scanning**: Bandit, Safety dependency checks
- **Documentation**: Auto-builds and deploys docs
- **Performance**: Tracks performance regressions

See `docs/workflows/` for detailed CI/CD setup.

## Release Process

### Version Management

Update version in:
- `pyproject.toml`
- `genrf/__init__.py`

### Creating Releases

```bash
# Tag release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# GitHub Actions will automatically:
# - Run full test suite
# - Build distributions
# - Create GitHub release
# - Publish to PyPI
```

## Contributing Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Getting Help

- **Documentation**: https://genrf-circuit.readthedocs.io
- **Issues**: https://github.com/yourusername/genRF-circuit-diffuser/issues
- **Discussions**: GitHub Discussions tab
- **Email**: daniel@example.com