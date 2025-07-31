# GitHub Actions Workflow Setup Guide

This document provides comprehensive instructions for setting up GitHub Actions workflows for the GenRF Circuit Diffuser project.

## Quick Setup

1. **Create `.github/workflows/` directory in your repository root**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy the comprehensive CI workflow**:
   ```bash
   cp docs/workflows/ci-comprehensive.yml .github/workflows/ci.yml
   ```

3. **Configure required secrets** in GitHub repository settings:
   - `PYPI_API_TOKEN`: For automated PyPI publishing
   - `CODECOV_TOKEN`: For code coverage reporting (optional)

## Workflow Components

### Core CI/CD Pipeline (`ci-comprehensive.yml`)

**Features:**
- Multi-Python version testing (3.8-3.11)
- Cross-platform testing (Linux, Windows, macOS)
- GPU-enabled ML model testing
- Comprehensive security scanning
- Performance regression testing
- SBOM generation
- Container security scanning
- Documentation deployment

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main`
- Weekly scheduled security scans
- Manual dispatch for GPU testing

### Security Scanning Components

1. **Static Analysis Security Testing (SAST)**:
   - Bandit for Python security issues
   - MyPy for type safety
   - Trivy for vulnerability scanning

2. **Dependency Scanning**:
   - Safety for Python package vulnerabilities
   - SBOM generation with CycloneDX
   - Automated security advisories

3. **Container Security**:
   - Trivy container image scanning
   - Multi-stage build optimization
   - Base image vulnerability assessment

### Performance Monitoring

1. **Benchmark Testing**:
   - pytest-benchmark for performance regression
   - Automated alerts on performance degradation
   - Historical performance tracking

2. **ML Model Validation**:
   - Model architecture validation
   - Performance regression testing
   - Memory usage profiling

## Required Repository Setup

### 1. Secrets Configuration

Add these secrets in GitHub repository settings:

```bash
# Repository Settings > Secrets and variables > Actions

PYPI_API_TOKEN        # PyPI publishing token
CODECOV_TOKEN         # Codecov integration (optional)
```

### 2. Branch Protection Rules

Configure branch protection for `main`:

```yaml
# Repository Settings > Branches > Add rule

Branch name pattern: main
Require status checks before merging: ✓
Require branches to be up to date: ✓
Status checks: 
  - quality
  - test (ubuntu-latest, 3.10)
  - security
  - docker
Require pull request reviews: ✓
Dismiss stale reviews: ✓
Require review from CODEOWNERS: ✓
```

### 3. GitHub Pages Setup

For documentation deployment:

1. Go to Repository Settings > Pages
2. Source: GitHub Actions
3. Build and deployment: GitHub Actions

### 4. GPU Runner Setup (Optional)

For ML model testing with GPU:

1. Set up self-hosted runner with GPU
2. Label runner with `gpu` tag
3. Ensure CUDA toolkit is installed

## Workflow Customization

### Environment Variables

Customize in workflow file:

```yaml
env:
  PYTHON_DEFAULT: '3.10'      # Default Python version
  PYTORCH_VERSION: '1.12.0'   # PyTorch version for GPU tests
```

### Matrix Strategy Adjustment

Modify test matrix based on your needs:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11']
    exclude:
      # Add exclusions to reduce CI time
```

### Security Scan Configuration

Adjust security thresholds:

```yaml
# Trivy configuration
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    severity: 'CRITICAL,HIGH'  # Adjust severity levels
    exit-code: '1'             # Fail on findings
```

## Advanced Features

### 1. Automated Dependency Updates

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 2. Issue Templates

Create `.github/ISSUE_TEMPLATE/`:

- `bug_report.yml`: Bug report template
- `feature_request.yml`: Feature request template
- `security_report.yml`: Security vulnerability template

### 3. Pull Request Templates

Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass

## Security
- [ ] Security scan passes
- [ ] No sensitive data exposed
```

## Monitoring and Alerts

### 1. GitHub Notifications

Configure notifications for:
- Failed builds
- Security alerts
- Performance regressions

### 2. Slack Integration (Optional)

Add Slack webhook for build notifications:

```yaml
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: always()
```

### 3. Email Alerts

Configure email notifications in repository settings for:
- Security advisories
- Dependabot alerts
- Workflow failures

## Troubleshooting

### Common Issues

1. **GPU Runner Not Available**:
   - Use `if: contains(github.event.pull_request.labels.*.name, 'gpu-test')`
   - Skip GPU tests if runner unavailable

2. **SPICE Installation Failures**:
   - Use containerized SPICE environment
   - Cache installation steps

3. **Performance Test Flakiness**:
   - Increase timeout values
   - Use multiple runs and averages

### Debug Commands

```bash
# Local workflow testing with act
act -j test

# Workflow syntax validation
gh workflow validate .github/workflows/ci.yml

# Manual workflow dispatch
gh workflow run ci.yml
```

## Best Practices

### 1. Workflow Optimization

- Use caching for dependencies
- Parallelize independent jobs
- Fail fast on critical errors
- Use matrix exclusions wisely

### 2. Security

- Never commit secrets to repository
- Use least privilege principle
- Regular security scan reviews
- Monitor dependency alerts

### 3. Performance

- Cache Docker layers
- Use job dependencies efficiently
- Skip redundant tests when possible
- Monitor workflow execution time

### 4. Maintenance

- Regular workflow updates
- Dependency version pinning
- Documentation updates
- Performance baseline updates

## Integration with External Tools

### 1. Codecov Integration

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests
    fail_ci_if_error: true
```

### 2. SonarCloud Integration

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### 3. SLSA Provenance

```yaml
- name: Generate SLSA provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
  with:
    base64-subjects: ${{ steps.hash.outputs.hashes }}
```

This comprehensive workflow setup provides enterprise-grade CI/CD capabilities while maintaining security and performance standards suitable for ML/AI projects in production environments.