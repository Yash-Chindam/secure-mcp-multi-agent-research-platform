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
- An MCP capability registry whose discovery reveals only what the caller may see.
- A governed gateway that authorizes, meters, circuit-breaks and audits every tool call.
- Enforced boundaries for the web, filesystem, PostgreSQL, GitHub and sandbox services.
- A FastMCP web research server the gateway drives over a real MCP round trip.
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

Capability discovery from section 8 is enforced by `research_platform.mcp.registry`. A capability
is revealed only when the caller's tenant, data clearance and role all permit it, and execution is
narrower than visibility so the planner reads tool metadata without ever running a tool. A
capability the caller cannot see is reported as missing rather than forbidden, so probing cannot
enumerate another tenant's tools.

```powershell
curl.exe -H "X-Tenant-ID: acme" -H "X-Requester-ID: user-1" `
  -H "X-Clearance: internal" http://127.0.0.1:8000/api/v1/capabilities
```

### The invocation gateway

Nothing reaches an MCP server except through `research_platform.mcp.gateway`. For every call it
resolves the capability against the caller, re-evaluates policy rather than trusting the earlier
discovery result, requires a digest-bound reviewer approval for sensitive capabilities, reserves
the job's tool call, consults the server circuit, and only then executes. Every outcome, including
every refusal, produces an audit record whose arguments are redacted.

Results come back as `UntrustedContent`: bounded by the capability's own size limit, stripped of
the zero-width characters used to hide text from a reviewer, and flagged when the source appears
to be steering the agent. Flagged text is preserved rather than rewritten, so the caller decides
whether to quarantine it instead of trusting content that merely looks clean.

A budget or circuit refusal is recorded as an authorized call that was stopped, never as a
permission denial, so the security metric in section 13 stays meaningful.

### Service boundaries

Each service in section 8 declares its own boundary, enforced inside the server as well as at
the gateway, so a future direct client inherits it:

| Service | Boundary |
|---|---|
| Web research | HTTPS-only allowlisted domains, standard port, no embedded credentials; private, loopback, link-local and cloud metadata addresses refused; sliding-window rate limit per tenant and host |
| Filesystem | Per-tenant absolute workspace root, relative paths only, readable document types, confinement re-checked after symlink resolution |
| PostgreSQL | One SELECT or WITH statement, comments and string literals stripped before keyword checks, every table schema-qualified to the caller's own schema, row limit tightened |
| GitHub | Per-tenant `owner/name` allowlist, ordinary branch and tag refs only |
| Python analysis | Network refused by construction, CPU, memory, wall-clock and output ceilings, submitted code screened for withheld capabilities |

The SQL parser and the sandbox code screen are defence in depth, not the guarantee. The database
role must be read-only with row-level security, and the container isolation is what actually
contains a calculation — Python cannot be made safe by inspection.

## Next milestones

1. FastMCP servers for the filesystem, PostgreSQL, GitHub and sandbox services.
2. OPA policy enforcement for tenant and role decisions.
3. Keycloak token verification at the API boundary.
4. CrewAI agents and Temporal durable execution.
5. OpenTelemetry tracing and the evaluation harness.
