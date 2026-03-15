# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (main branch) | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email the maintainer directly or use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) feature.
3. Include a detailed description of the vulnerability, steps to reproduce, and potential impact.

### What to Expect

- Acknowledgment of your report within 48 hours.
- An assessment of the vulnerability and an estimated timeline for a fix.
- Credit in the fix commit (unless you prefer to remain anonymous).

## Scope

This project is a research/educational deep learning codebase. Security considerations include:

- **Dependency vulnerabilities**: Issues in PyTorch, NumPy, or other dependencies.
- **Model deserialization**: Risks associated with loading untrusted `.pt` weight files (pickle-based).
- **Configuration injection**: Malformed YAML configurations that could cause unexpected behavior.

## Best Practices for Users

- Only load model weights (`.pt` files) from trusted sources.
- Keep dependencies up to date by periodically running `pip install --upgrade -r requirements.txt`.
- Review YAML configuration files before use, especially from third-party sources.
