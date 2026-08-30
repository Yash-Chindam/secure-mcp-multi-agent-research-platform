# Secure MCP Multi-Agent Research Platform

A typed Python foundation for the governed research system described in
[`01-secure-mcp-multi-agent-research-platform.md`](./01-secure-mcp-multi-agent-research-platform.md).

## What is implemented

The foundation provides:

- Explicit workflow state transitions that reject invalid stage skipping.
- Evidence records with SHA-256 validation.
- Claim schemas that require supporting evidence.
- Tenant-isolated job and evidence services.
- Task decomposition with dependency ordering, retry budgets and cycle rejection.
- Tool invocation auditing with argument redaction and a retry-aware error taxonomy.
- Reviewer approvals bound to one exact server, capability, resource and argument digest.
- A typed FastAPI boundary for research jobs and evidence.
- A requester interface for creating and listing assignments.
- Strict typing, unit tests, API integration tests, and Playwright browser coverage.
- A container image build verified on every pull request.
- Dependency, labeling, and merge automation for the repository.

The identity headers in this milestone are a trusted-proxy seam for local development. They are
not production authentication. Keycloak token verification and OPA policy enforcement remain
future security milestones.

## Develop locally

Requires Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\uvicorn.exe research_platform.main:app --reload
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\pytest.exe tests\unit tests\integration --cov
.\.venv\Scripts\playwright.exe install chromium
.\.venv\Scripts\pytest.exe tests\e2e -m e2e --browser chromium
```

Open <http://127.0.0.1:8000>. API documentation is available at
<http://127.0.0.1:8000/docs>.

## Run the container

```powershell
docker build -t secure-mcp-research-platform:local .
docker run --rm -p 8000:8000 secure-mcp-research-platform:local
```

The image installs the package from a wheel built in a separate stage, runs as an unprivileged
`app` user, and reports health through `/health`. Continuous integration rebuilds the image on
every pull request without publishing it to any registry.

## Repository automation

- Dependabot opens weekly grouped updates for pip, GitHub Actions, and the Docker base image.
- A labeler applies area labels to every pull request from its changed paths.
- Pull requests merge automatically once linting, typing, unit, integration, Playwright, and
  container jobs all pass. Merge commits are used so each branch keeps its individual commits.
  Prefix a branch with `no-automerge/` to opt out.
- Non-major Dependabot updates merge on the same evidence.

Pull requests that change files under `.github/workflows/` are merged by a maintainer, because
the Actions token is not permitted to update workflow definitions.

## Design specification coverage

The information model in section 10 of the design specification is implemented in full:
`ResearchJob`, `ResearchTask`, `EvidenceRecord`, `Finding`, `ToolInvocation` and `ApprovalRequest`.

## Next milestones

1. MCP capability registry and the FastMCP research servers.
2. OPA policy enforcement for tenant and role decisions.
3. Keycloak token verification at the API boundary.
