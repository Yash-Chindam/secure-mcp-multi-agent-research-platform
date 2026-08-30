# Secure MCP Multi-Agent Research Platform

A typed Python foundation for the governed research system described in
[`01-secure-mcp-multi-agent-research-platform.md`](./01-secure-mcp-multi-agent-research-platform.md).

## What is implemented

The foundation provides:

- Explicit workflow state transitions that reject invalid stage skipping.
- Evidence records with SHA-256 validation.
- Claim schemas that require supporting evidence.
- Tenant-isolated job and evidence services.
- Strict typing, formatting, linting, and domain unit tests.

## Develop locally

Requires Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\pytest.exe tests\unit --cov
```

## Next milestones

1. FastAPI endpoints and integration tests.
2. A requester UI and Playwright end-to-end tests.
3. Container build verification and repository automation.
