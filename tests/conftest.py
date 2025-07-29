"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_design_spec():
    """Sample design specification for testing."""
    return {
        "circuit_type": "LNA",
        "frequency": 2.4e9,
        "gain_min": 15,
        "nf_max": 1.5,
        "power_max": 10e-3,
        "technology": "TSMC65nm"
    }