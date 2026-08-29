# Secure MCP Multi-Agent Research Platform

A governed, tenant-aware foundation for the production research system described in
[`01-secure-mcp-multi-agent-research-platform.md`](./01-secure-mcp-multi-agent-research-platform.md).

## What is implemented

The first vertical slice provides:

- A typed FastAPI API for creating, listing, reading, and transitioning research jobs.
- Tenant isolation at every repository lookup; cross-tenant access returns `404`.
- Explicit workflow state transitions that reject invalid stage skipping.
- Evidence records with SHA-256 validation and claim schemas that require citations.
- A minimal requester UI for creating and viewing research assignments.
- Unit tests, API integration tests, and a Chromium Playwright end-to-end test.
- Container build verification, Dependabot, path-based PR labeling, and CI-gated auto-merge.

The identity headers in this slice are a trusted-proxy seam for local development. They are
not production authentication. Keycloak token verification and OPA policy enforcement are the
next security milestone.

## Run locally

Requires Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\uvicorn.exe research_platform.main:app --reload
```

Open <http://127.0.0.1:8000>. API documentation is available at
<http://127.0.0.1:8000/docs>.

## Test locally

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\pytest.exe tests\unit tests\integration --cov
.\.venv\Scripts\playwright.exe install chromium
.\.venv\Scripts\pytest.exe tests\e2e -m e2e --browser chromium
```

## Pull request automation

Every non-draft PR from a branch in this repository is configured for squash auto-merge after
linting, strict typing, unit tests, integration tests, Playwright E2E, and the container build
all pass. Prefix a branch with `no-automerge/` when a manual merge is required. Dependabot
minor and patch updates are approved and queued for auto-merge; major updates remain manual.

Repository settings must enable GitHub auto-merge and require the CI jobs on the default branch.
The pipeline builds the deployable container but does not publish it until a registry and
deployment environment are explicitly configured.

## Next milestones

1. PostgreSQL persistence and migrations with tenant row-level security.
2. Keycloak OIDC validation and OPA authorization decisions.
3. MCP capability registry and policy-filtered tool discovery.
4. Temporal workflow plus CrewAI planner, researcher, analyst, critic, and reporter stages.
5. Approval signals, evidence artifacts, observability, and failure-recovery tests.
