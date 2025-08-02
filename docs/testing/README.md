# Testing Documentation

This document provides comprehensive information about the testing strategy and infrastructure for the GenRF Circuit Diffuser project.

## Testing Strategy

Our testing approach follows a multi-layered strategy to ensure robust, reliable circuit generation:

### Testing Pyramid

```
    E2E Tests (Few)
   ╱─────────────────╲
  ╱   Integration     ╲
 ╱     Tests (Some)    ╲  
╱─────────────────────────╲
│   Unit Tests (Many)      │
╰─────────────────────────╯
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Core algorithms, utility functions, data structures
- **Speed**: Fast (<1s per test)
- **Coverage Target**: >95%

### 2. Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and interfaces
- **Scope**: SPICE integration, model loading, data flow
- **Speed**: Medium (1-30s per test)
- **Coverage Target**: >80%

### 3. End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Scope**: Full pipeline from specification to circuit export
- **Speed**: Slow (30s-5min per test)
- **Coverage Target**: Critical paths only

### 4. Performance Tests (`tests/performance/`)
- **Purpose**: Validate performance characteristics
- **Scope**: Generation speed, memory usage, scalability
- **Speed**: Variable (seconds to hours)
- **Metrics**: Latency, throughput, resource utilization

### 5. Security Tests (`tests/test_security.py`)
- **Purpose**: Identify security vulnerabilities
- **Scope**: Input validation, dependency scanning, secrets
- **Speed**: Fast to medium
- **Tools**: Bandit, safety, custom validators

## Test Markers

Use pytest markers to categorize and selectively run tests:

```bash
# Run only fast unit tests
pytest -m "not slow and not integration"

# Run integration tests
pytest -m integration

# Run tests for specific circuit types
pytest -m lna
pytest -m mixer

# Run tests for specific technologies
pytest -m tsmc65
pytest -m gf22

# Skip GPU tests if no GPU available
pytest -m "not gpu"

# Skip SPICE tests if no SPICE engine
pytest -m "not spice"
```

### Available Markers

| Marker | Description |
|--------|-------------|
| `benchmark` | Performance benchmarking tests |
| `integration` | Integration tests |
| `spice` | Tests requiring SPICE simulation |
| `slow` | Tests taking >30 seconds |
| `gpu` | Tests requiring GPU/CUDA |
| `lna` | LNA-specific tests |
| `mixer` | Mixer-specific tests |
| `vco` | VCO-specific tests |
| `pa` | Power amplifier tests |
| `filter` | Filter-specific tests |
| `tsmc65` | TSMC 65nm technology tests |
| `gf22` | GlobalFoundries 22nm tests |
| `sky130` | SkyWater 130nm tests |

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genrf

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::test_circuit_generation

# Run tests in parallel
pytest -n auto
```

### Test Selection

```bash
# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/

# Run tests matching pattern
pytest -k "test_lna"

# Run failed tests from last run
pytest --lf
```

### Test Configuration

The main test configuration is in `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = 
    --strict-markers
    --cov=genrf
    --cov-report=html:htmlcov
    --cov-fail-under=80
    --durations=10
```

## Test Fixtures

### Circuit Specifications

Common circuit specifications are available as fixtures:

```python
def test_lna_generation(lna_spec):
    """Test using predefined LNA specification."""
    assert lna_spec["circuit_type"] == "LNA"
    assert lna_spec["frequency"] == 2.4e9
```

### Mock Components

Mock SPICE engines and other external dependencies:

```python
def test_simulation(mock_spice_engine):
    """Test with mocked SPICE engine."""
    results = mock_spice_engine.simulate(netlist)
    assert results["converged"]
```

### Test Data

Fixtures provide realistic test data:

```python
def test_optimization(simulation_results):
    """Test with sample simulation results."""
    gain = max(simulation_results["s21_mag"])
    assert gain > 15  # Meet gain requirement
```

## Continuous Integration

### GitHub Actions Workflow

Our CI pipeline runs:

1. **Linting and Formatting**
   - Black code formatting
   - Flake8 linting
   - MyPy type checking

2. **Testing Matrix**
   - Python 3.8, 3.9, 3.10, 3.11
   - Multiple operating systems
   - With/without optional dependencies

3. **Coverage Reporting**
   - Upload to Codecov
   - Fail if coverage drops below threshold

4. **Performance Regression**
   - Run benchmark tests
   - Compare against baseline

### Local Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Writing Tests

### Test Naming Convention

```python
def test_should_generate_lna_when_valid_spec():
    """Test function names should describe expected behavior."""
    pass

class TestCircuitGenerator:
    """Test classes group related functionality."""
    
    def test_should_validate_input_specification(self):
        pass
    
    def test_should_raise_error_for_invalid_frequency(self):
        pass
```

### Test Structure (Arrange-Act-Assert)

```python
def test_circuit_generation():
    # Arrange
    spec = {"circuit_type": "LNA", "frequency": 2.4e9}
    generator = CircuitGenerator()
    
    # Act
    circuit = generator.generate(spec)
    
    # Assert
    assert circuit is not None
    assert circuit.gain > spec.get("gain_min", 0)
```

### Parameterized Tests

```python
@pytest.mark.parametrize("circuit_type,expected_topology", [
    ("LNA", "common_source"),
    ("Mixer", "gilbert_cell"),
    ("VCO", "lc_tank"),
])
def test_topology_selection(circuit_type, expected_topology):
    spec = {"circuit_type": circuit_type}
    topology = select_topology(spec)
    assert topology == expected_topology
```

## Performance Testing

### Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_generation_speed(benchmark, lna_spec):
    """Benchmark circuit generation speed."""
    result = benchmark(generate_circuit, lna_spec)
    assert result is not None

@pytest.mark.benchmark
def test_memory_usage():
    """Test memory consumption during generation."""
    import tracemalloc
    
    tracemalloc.start()
    generate_large_batch()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory usage is reasonable
    assert peak < 1024 * 1024 * 1024  # 1GB limit
```

### Load Testing

```python
@pytest.mark.slow
def test_concurrent_generation():
    """Test system under concurrent load."""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_circuit, spec)
            for spec in circuit_specs
        ]
        
        results = [future.result() for future in futures]
        assert all(result is not None for result in results)
```

## Test Data Management

### Test Datasets

Large test datasets are stored separately:

```
tests/
├── fixtures/
│   ├── circuit_data.py     # Programmatic fixtures
│   ├── netlists/           # Sample SPICE netlists
│   ├── models/             # Test model files
│   └── results/            # Reference simulation results
```

### Data Generation

```python
def generate_test_data():
    """Generate synthetic test data for development."""
    import numpy as np
    
    # Create synthetic S-parameters
    freq = np.logspace(9, 11, 100)  # 1-100 GHz
    s21 = 15 + 5 * np.random.normal(0, 0.1, len(freq))
    
    return {"frequency": freq, "s21_mag": s21}
```

## Troubleshooting

### Common Issues

1. **SPICE Engine Not Found**
   ```bash
   # Install ngspice
   conda install -c conda-forge ngspice
   
   # Or skip SPICE tests
   pytest -m "not spice"
   ```

2. **GPU Tests Failing**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Skip GPU tests
   pytest -m "not gpu"
   ```

3. **Slow Test Performance**
   ```bash
   # Run tests in parallel
   pytest -n auto
   
   # Skip slow tests
   pytest -m "not slow"
   ```

### Debug Mode

```bash
# Run with detailed output
pytest -v --tb=long

# Drop into debugger on failure
pytest --pdb

# Run with profiling
pytest --profile

# Show test durations
pytest --durations=10
```

## Test Coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=genrf --cov-report=html

# View in browser
open htmlcov/index.html

# Generate terminal report
pytest --cov=genrf --cov-report=term-missing
```

### Coverage Targets

| Component | Target | Current |
|-----------|---------|---------|
| Core Logic | 95% | TBD |
| API Endpoints | 90% | TBD |
| CLI Interface | 85% | TBD |
| Integration | 80% | TBD |
| Overall | 85% | TBD |

## Contributing Tests

### Test Requirements

1. **Every new feature must include tests**
2. **Bug fixes must include regression tests**
3. **Tests must be fast and reliable**
4. **Use appropriate test markers**
5. **Follow naming conventions**

### Review Checklist

- [ ] Tests cover happy path and error cases
- [ ] Appropriate fixtures and mocks used
- [ ] Tests are independent and isolated
- [ ] Performance implications considered
- [ ] Documentation updated if needed

For more information, see the [Contributing Guide](../../CONTRIBUTING.md).