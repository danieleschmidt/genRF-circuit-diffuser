# Contributing to GenRF Circuit Diffuser

We welcome contributions to improve RF circuit generation! This document provides guidelines for contributors.

## Development Setup

1. **Clone and install**:
   ```bash
   git clone https://github.com/yourusername/genRF-circuit-diffuser.git
   cd genRF-circuit-diffuser
   make install-dev
   ```

2. **Run tests**:
   ```bash
   make test
   ```

3. **Code formatting**:
   ```bash
   make format
   make lint
   ```

## Contribution Areas

### High Priority
- Additional PDK support (TSMC, GlobalFoundries, Intel)
- mmWave circuit types (>40 GHz)
- EM co-simulation integration
- Layout generation algorithms

### Medium Priority
- Advanced SPICE engine support (Spectre, HSPICE)
- Quantum-inspired optimization algorithms
- Neural ODE circuit models
- Additional export formats

## Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**:
   - Follow existing code style (Black + isort)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test thoroughly**:
   ```bash
   make check  # Runs lint + tests
   ```

4. **Submit pull request**:
   - Clear description of changes
   - Link to related issues
   - Include performance benchmarks if applicable

## Code Style

- **Formatting**: Use Black with 88-character line length
- **Imports**: Use isort with Black profile
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style docstrings
- **Tests**: Pytest with >90% coverage target

## Testing Guidelines

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test SPICE simulation integration
- **Performance tests**: Benchmark generation speed and quality
- **Regression tests**: Ensure backward compatibility

## Documentation

- Update API documentation for new classes/methods
- Add tutorial sections for major features
- Include code examples in docstrings
- Update README.md for user-facing changes

## Issue Reporting

When reporting bugs, please include:
- Python version and dependency versions
- Minimal reproduction example
- Expected vs actual behavior
- System information (OS, SPICE engine)

## Performance Considerations

- Profile code changes affecting generation speed
- Include benchmark comparisons in PR descriptions
- Consider memory usage for large design spaces
- Optimize SPICE simulation calls

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.