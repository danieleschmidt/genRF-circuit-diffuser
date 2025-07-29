# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities via email to: security@example.com

You should receive a response within 48 hours. If the issue is confirmed, we will:

1. Work to understand the scope and severity
2. Develop and test a fix
3. Prepare a security advisory
4. Release the fix and advisory simultaneously

## Security Considerations

### Circuit Generation
- Generated circuits should be validated before production use
- SPICE simulations provide design-time verification only
- Silicon validation is required for production deployment

### Data Handling  
- Design specifications may contain proprietary information
- Model checkpoints should be stored securely
- Generated circuits may contain intellectual property

### Dependencies
- Regular security scanning of Python dependencies
- Pin dependency versions in production
- Monitor for CVEs in PyTorch and scientific computing packages

### SPICE Integration
- Validate SPICE netlist inputs to prevent injection attacks
- Sandbox SPICE simulation environments
- Limit simulation resources to prevent DoS

## Responsible Disclosure

We appreciate security researchers who help improve our security posture. For responsible disclosure:

1. Allow reasonable time for investigation and remediation
2. Do not access or modify user data without permission
3. Do not perform actions that could harm system availability
4. Report vulnerabilities as soon as possible

## Security Updates

Security updates will be released as patch versions and announced via:
- GitHub Security Advisories
- Project mailing list
- Release notes with CVE information