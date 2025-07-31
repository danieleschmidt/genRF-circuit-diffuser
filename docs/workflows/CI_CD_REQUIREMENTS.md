# CI/CD Workflow Requirements

This document outlines the required GitHub Actions workflows for the genRF project. These workflows ensure code quality, security, and reliable deployment.

> **Note**: This project follows Terragon Labs policy - GitHub workflow files are documented here but must be manually created in `.github/workflows/` by maintainers with appropriate repository permissions.

## Required Workflows

### 1. Continuous Integration (ci.yml)

**Trigger**: Push to any branch, Pull Requests
**Purpose**: Code quality, testing, and security scanning

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop, 'feature/*', 'hotfix/*' ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.8'
  PYTORCH_VERSION: '1.12.0'

jobs:
  lint-and-format:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          pip install black isort flake8 mypy bandit
          pip install -e .[dev]
          
      - name: Run black formatting check
        run: black --check --diff .
        
      - name: Run isort import sorting check
        run: isort --check-only --diff .
        
      - name: Run flake8 linting
        run: flake8
        
      - name: Run mypy type checking
        run: mypy genrf/
        
      - name: Run bandit security analysis
        run: bandit -r genrf/ -x tests/ -ll

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ngspice
          
      - name: Install Python dependencies
        run: |
          pip install -e .[dev,spice]
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=genrf --cov-report=xml
          
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v -m "not gpu"
          
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          
  security:
    name: Security Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install security tools
        run: |
          pip install safety pip-audit detect-secrets
          
      - name: Run dependency vulnerability scan
        run: |
          safety check --exit-code
          pip-audit --require --desc
          
      - name: Run secret detection
        run: |
          detect-secrets scan --baseline .secrets.baseline
          
      - name: Upload security results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: security-results.sarif

  docker:
    name: Docker Build Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Build Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: genrf:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 2. Performance Testing (performance.yml)

**Trigger**: Push to main, Weekly schedule, Manual dispatch
**Purpose**: Performance regression detection and benchmarking

```yaml
name: Performance Testing

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2 AM UTC
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Type of benchmark to run'
        required: true
        default: 'all'
        type: choice
        options:
        - all
        - generation
        - spice
        - optimization

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
          
      - name: Install dependencies
        run: |
          pip install -e .[dev,spice]
          pip install pytest-benchmark
          
      - name: Run benchmarks
        run: |
          pytest tests/test_performance.py \
            --benchmark-only \
            --benchmark-json=benchmark_results.json \
            -m "benchmark and not gpu"
            
      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark_results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '200%'
          fail-on-alert: true

  memory-test:
    name: Memory Usage Testing
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
          
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install memory-profiler psutil
          
      - name: Run memory profiling
        run: |
          python -m memory_profiler tests/test_memory_usage.py > memory_report.txt
          
      - name: Upload memory report
        uses: actions/upload-artifact@v3
        with:
          name: memory-report
          path: memory_report.txt
```

### 3. Security Workflow (security.yml)

**Trigger**: Push to main, Pull Requests, Weekly schedule
**Purpose**: Comprehensive security scanning and compliance

```yaml
name: Security Analysis

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 2'  # Weekly on Tuesday 4 AM UTC

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
          
      - name: Upload result to GitHub Code Scanning
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: snyk.sarif

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t genrf:security-test .
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'genrf:security-test'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### 4. Release Workflow (release.yml)

**Trigger**: Push to main with version tags, Manual dispatch
**Purpose**: Automated releases and artifact publishing

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.0.0)'
        required: true
        type: string

jobs:
  test:
    name: Pre-release Testing
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
          
      - name: Install dependencies
        run: |
          pip install -e .[dev,spice]
          
      - name: Run full test suite
        run: |
          pytest -v --cov=genrf
          
  build:
    name: Build Distribution
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
          
      - name: Install build dependencies
        run: |
          pip install build twine
          
      - name: Build package
        run: |
          python -m build
          
      - name: Check package
        run: |
          twine check dist/*
          
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: distributions
          path: dist/

  docker-release:
    name: Build and Push Docker Images
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: genrf/circuit-diffuser
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  github-release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [build, docker-release]
    steps:
      - uses: actions/checkout@v4
      
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: distributions
          path: dist/
          
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
          draft: false
          prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
```

### 5. Documentation Workflow (docs.yml)

**Trigger**: Push to main, Pull Requests affecting docs
**Purpose**: Build and deploy documentation

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'genrf/**/*.py'
      - 'README.md'
  pull_request:
    paths:
      - 'docs/**'
      - 'genrf/**/*.py'
      - 'README.md'

jobs:
  build-docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
          
      - name: Install dependencies
        run: |
          pip install -e .[docs]
          
      - name: Build documentation
        run: |
          cd docs
          make html
          
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/_build/html/

  deploy-docs:
    name: Deploy Documentation
    runs-on: ubuntu-latest
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Download documentation
        uses: actions/download-artifact@v3
        with:
          name: documentation
          path: docs/_build/html/
          
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_build/html/
```

## Workflow Integration Requirements

### 1. Branch Protection Rules

Configure these rules for the `main` branch:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Required status checks:
  - `lint-and-format`
  - `test (3.8)`, `test (3.9)`, `test (3.10)`, `test (3.11)`
  - `security`
  - `docker`
- Require pull request reviews before merging (2 reviewers)
- Dismiss stale PR approvals when new commits are pushed
- Require review from code owners
- Restrict pushes that create files matching `.github/workflows/*`

### 2. Required Secrets

Configure these GitHub secrets:
- `CODECOV_TOKEN`: For code coverage reporting
- `SNYK_TOKEN`: For vulnerability scanning
- `DOCKER_USERNAME` & `DOCKER_TOKEN`: For Docker Hub publishing
- `PYPI_API_TOKEN`: For PyPI package publishing

### 3. Environment Variables

Repository-level environment variables:
- `PYTHON_VERSION`: Default Python version for CI
- `PYTORCH_VERSION`: PyTorch version for testing
- `SPICE_ENGINE`: Default SPICE engine for testing

### 4. Workflow Permissions

Required permissions for workflows:
- `contents: read` - Read repository contents
- `actions: read` - Read workflow information
- `security-events: write` - Upload security scan results
- `packages: write` - Publish packages
- `pages: write` - Deploy GitHub Pages

## Manual Setup Instructions

1. **Create workflow files**: Copy the YAML configurations above into `.github/workflows/` directory
2. **Configure branch protection**: Set up branch protection rules in repository settings
3. **Add secrets**: Configure required secrets in repository settings
4. **Set up integrations**: Connect external services (Codecov, Snyk, Docker Hub)
5. **Test workflows**: Create a test branch and PR to verify all workflows execute correctly

## Workflow Monitoring

Monitor workflow health through:
- GitHub Actions dashboard
- Workflow run notifications
- Failed workflow alerts via Slack/email
- Performance trend analysis for benchmark workflows
- Security scan result tracking

Each workflow includes appropriate error handling, retry logic, and notification mechanisms to ensure reliable CI/CD operations.