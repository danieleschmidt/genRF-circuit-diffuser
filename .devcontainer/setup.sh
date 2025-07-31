#!/bin/bash
# Post-create setup script for GenRF Circuit Diffuser development container

set -e

echo "🚀 Setting up GenRF Circuit Diffuser development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Make sure you're in the project root."
    exit 1
fi

# Install the project in development mode
print_status "Installing GenRF Circuit Diffuser in development mode..."
if pip install -e ".[dev,spice,docs]"; then
    print_success "Project installed successfully"
else
    print_error "Failed to install project"
    exit 1
fi

# Install pre-commit hooks
print_status "Setting up pre-commit hooks..."
if pre-commit install; then
    print_success "Pre-commit hooks installed"
else
    print_warning "Failed to install pre-commit hooks"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p {outputs,generated_circuits,spice_netlists,logs,experiments,models}
print_success "Project directories created"

# Set up SPICE environment
print_status "Configuring SPICE environment..."
if command -v ngspice >/dev/null 2>&1; then
    echo "export SPICE_LIB_DIR=/usr/share/ngspice" >> ~/.bashrc
    echo "export SPICE_EXEC_DIR=/usr/bin" >> ~/.bashrc
    print_success "SPICE environment configured (NgSpice found)"
else
    print_warning "NgSpice not found. SPICE simulation may not work."
fi

# Set up Python path
print_status "Configuring Python environment..."
echo "export PYTHONPATH=\$PYTHONPATH:/workspaces/genrf-circuit-diffuser" >> ~/.bashrc
echo "export GENRF_HOME=/workspaces/genrf-circuit-diffuser" >> ~/.bashrc
echo "export GENRF_ENV=development" >> ~/.bashrc

# Create a simple test to verify installation
print_status "Running installation verification..."
cat > /tmp/test_install.py << 'EOF'
#!/usr/bin/env python3
"""Quick installation verification script."""

import sys
import importlib

def test_import(module_name, description):
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False

def main():
    print("🔍 Verifying GenRF Circuit Diffuser installation...\n")
    
    tests = [
        ("genrf", "GenRF main package"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("plotly", "Plotly"),
        ("yaml", "PyYAML"),
        ("pytest", "Pytest"),
        ("black", "Black formatter"),
        ("isort", "isort"),
        ("mypy", "MyPy type checker"),
    ]
    
    passed = 0
    total = len(tests)
    
    for module, description in tests:
        if test_import(module, description):
            passed += 1
    
    print(f"\n✅ Installation verification: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dependencies are properly installed!")
        return 0
    else:
        print("⚠️  Some dependencies are missing. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

if python /tmp/test_install.py; then
    print_success "Installation verification passed"
else
    print_warning "Some issues found during verification"
fi

# Set up Jupyter notebook extensions (if running interactively)
if [ -t 1 ]; then
    print_status "Setting up Jupyter extensions..."
    jupyter notebook --generate-config 2>/dev/null || true
    print_success "Jupyter configured"
fi

# Create a development configuration file
print_status "Creating development configuration..."
cat > .genrf_dev_config << 'EOF'
# GenRF Circuit Diffuser Development Configuration
# This file is automatically sourced in development containers

# Development settings
export GENRF_DEBUG=1
export GENRF_LOG_LEVEL=DEBUG
export GENRF_CACHE_DIR=/opt/genrf/cache
export GENRF_MODEL_DIR=/opt/genrf/models
export GENRF_OUTPUT_DIR=/opt/genrf/outputs

# SPICE simulation settings
export SPICE_ENGINE=ngspice
export SPICE_TIMEOUT=300

# ML model settings
export TORCH_HOME=/opt/genrf/cache/torch
export TRANSFORMERS_CACHE=/opt/genrf/cache/transformers

# Performance settings
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
EOF

# Add development config to bashrc
echo "source /workspaces/genrf-circuit-diffuser/.genrf_dev_config" >> ~/.bashrc

print_success "Development configuration created"

# Final setup message
print_success "🎉 GenRF Circuit Diffuser development environment setup complete!"

echo ""
echo "📝 Quick start commands:"
echo "  • Run tests: pytest tests/"
echo "  • Format code: black ."
echo "  • Type check: mypy genrf/"
echo "  • Generate circuit: python -m genrf.cli generate --help"
echo "  • Build docs: cd docs && make html"
echo "  • Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "🔧 Development tools installed:"
echo "  • Pre-commit hooks for code quality"
echo "  • VS Code extensions configured"
echo "  • SPICE simulation environment"
echo "  • ML/AI development stack"
echo ""
echo "📚 Documentation: https://genrf-circuit.readthedocs.io"
echo "🐛 Issues: https://github.com/yourusername/genRF-circuit-diffuser/issues"
echo ""

# Clean up
rm -f /tmp/test_install.py

print_success "Setup script completed successfully! 🚀"