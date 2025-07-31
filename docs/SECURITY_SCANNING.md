# Security Scanning Guide

This document outlines security practices and scanning requirements for the genRF project.

## Overview

GenRF handles sensitive circuit designs and proprietary PDK information. Security scanning ensures:
- No secrets are committed to the repository
- Dependencies are free from known vulnerabilities  
- Generated circuit files don't contain sensitive information
- SPICE simulation data is properly sanitized

## Required Security Tools

### 1. Dependency Vulnerability Scanning

**Tool**: Safety + pip-audit

```bash
# Install tools
pip install safety pip-audit

# Scan for vulnerabilities  
safety check --json --output vulns.json
pip-audit --format=json --output=audit.json

# In CI: Fail on HIGH/CRITICAL vulnerabilities
safety check --continue-on-error --exit-code
```

**Configuration**: Add to `pyproject.toml`:
```toml
[tool.safety]
ignore = []
continue-on-vulnerabilities = false
full-report = true
```

### 2. Secret Detection

**Tool**: detect-secrets

```bash
# Install and initialize
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline

# Pre-commit integration already configured
# Manually scan:
detect-secrets audit .secrets.baseline
```

**Sensitive Patterns**:
- PDK license keys
- SPICE model files with proprietary parameters
- API keys for cloud simulation services
- Circuit parameters marked as confidential

### 3. SAST (Static Application Security Testing)

**Tool**: Bandit for Python security

```bash
# Run security analysis
bandit -r genrf/ -f json -o security-report.json

# Configuration in pyproject.toml excludes test files
# Focus on:
# - Command injection in SPICE calls
# - File path traversal in circuit exports
# - Unsafe deserialization of models
```

### 4. Container Security

**Tool**: Docker Scout / Trivy

```bash
# Scan Docker images
docker scout cves genrf:latest
# OR
trivy image genrf:latest --format json --output container-scan.json
```

## CI/CD Integration Requirements

### GitHub Actions Workflow Template

```yaml
name: Security Scanning
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Dependency Check
        run: |
          pip install safety pip-audit
          safety check --exit-code
          pip-audit --require --desc
          
      - name: Secret Scan
        run: |
          pip install detect-secrets
          detect-secrets scan --baseline .secrets.baseline
          
      - name: SAST Analysis  
        run: |
          pip install bandit
          bandit -r genrf/ -x tests/ -ll
          
      - name: Upload Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: security-results.sarif
```

## Circuit-Specific Security Considerations

### 1. PDK Protection
- Never commit technology files (`.lib`, `.tech`, `.lef`)
- Use environment variables for PDK paths
- Implement PDK fingerprinting to detect unauthorized usage

### 2. Generated Circuit Sanitization
```python
# Example: Sanitize exported circuits
def sanitize_netlist(netlist_content):
    # Remove proprietary model parameters
    # Anonymize node names containing sensitive info
    # Strip comments with design intent
    return cleaned_content
```

### 3. SPICE Simulation Security
- Sandbox SPICE execution using containers
- Validate all input netlists before simulation
- Scrub simulation logs of sensitive parameters

## Reporting Security Issues

See [SECURITY.md](../SECURITY.md) for vulnerability reporting procedures.

## Compliance Requirements

### Export Control (ITAR/EAR)
- Certain RF circuits may be export-controlled
- Implement geo-blocking for restricted regions
- Add export control warnings to generated designs

### Corporate PDK Licenses
- Validate PDK usage permissions before generation
- Log all PDK accesses for audit trails
- Implement usage quotas and rate limiting

## Security Metrics

Track these security health metrics:
- Time to fix HIGH/CRITICAL vulnerabilities: < 7 days
- Secret detection false positive rate: < 5%
- Dependency update coverage: > 90%
- Security test coverage: > 80%

## Emergency Response

**Security Incident Response Plan**:
1. **Detection**: Automated alerts for security violations
2. **Assessment**: Triage severity and impact scope  
3. **Containment**: Revoke access, rotate credentials
4. **Recovery**: Deploy patches, validate systems
5. **Lessons Learned**: Update procedures and tooling