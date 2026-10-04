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
- All five FastMCP research servers, driven by the gateway over real MCP round trips.
- Policy-as-code authorization in Rego, layered with the registry boundary and failing closed.
- OAuth access token verification against a Keycloak realm.
- CrewAI planner, researcher, analyst, critic and reporter agents on a durable Temporal
  workflow, with a reviewer checkpoint that suspends without holding a worker.
- A PostgreSQL system of record with forced row-level security per tenant.
- Evidence the platform verifies itself: an excerpt must appear in what a recorded tool
  call returned, and every later citation must name recorded evidence.
- Findings stored with their critic verdict and reviewer status.
- A published report (JSON and Markdown), provenance manifest and evidence bundle,
  exported to MinIO under a per-tenant prefix.
- Content-drift detection: cited sources are re-read at publication and flagged if the
  captured excerpt is gone, while the original evidence is kept.
- A typed FastAPI boundary for research jobs, evidence, findings, reports and audit trails.
- A requester interface for creating and listing assignments.
- Strict typing, unit tests, API integration tests, and Playwright browser coverage.
- A container image build verified on every pull request.
- Dependency, labeling, and merge automation for the repository.

## Identity

Two identity sources exist and they are mutually exclusive.

Set `RESEARCH_OIDC_ISSUER` and `RESEARCH_OIDC_AUDIENCE` and a verified bearer token becomes the
only accepted identity: the development headers stop being read at all, so a configured
deployment cannot be downgraded by sending one. Signing keys are fetched from the realm's JWKS
endpoint, and the accepted algorithms are an asymmetric allowlist so a token cannot present a
symmetric algorithm the verifier would then check with the realm's own public key.

Leave them unset and the `X-Tenant-ID`, `X-Requester-ID`, `X-Roles` and `X-Clearance` headers are
accepted as a local development seam. That is not production authentication, and `GET /health`
says which source is active rather than leaving an operator to assume tokens are being checked.

A realm role the platform does not recognize is dropped rather than interpreted generously, and an
absent clearance claim defaults to the least permissive class.

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

### Authorization

Authorization is decided by two layers that must both permit a call:

1. The **registry boundary** — the platform's own invariant, derived from the registered
   capability.
2. The **Open Policy Agent bundle** in [`policy/research/authz.rego`](./policy/research/authz.rego)
   — the organization's policy, consulted over HTTP when `RESEARCH_OPA_URL` is set.

Requiring both means a misconfigured bundle cannot widen access beyond the registered
capability, and a capability cannot override a policy that has narrowed it. The bundle also
carries a rule the catalogue cannot express: no agent may execute a side-effecting capability,
whatever its `allowed_agents` list says — only a human role may cause an external side effect.

The policy client fails closed. A timeout, transport error, error status or unreadable response
all refuse the call, because an authorization service that is unreachable must never be
equivalent to one that said yes. Argument *names* are sent to the policy; argument *values* are
not, so asking for a decision does not copy sensitive data into a second service and its logs.

`GET /health` reports which layers are active, so running without a policy service is stated
rather than silently assumed.

```powershell
$env:RESEARCH_OPA_URL = "http://localhost:8181"
docker run --rm -p 8181:8181 -v "${PWD}/policy:/policy" openpolicyagent/opa:1.11.0 `
  run --server --addr :8181 /policy
```

Policy tests run in CI with `opa check --strict` and `opa test`.

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

All five servers are built by `research_platform.mcp.servers.deployment.build_servers`, which
registers a service only when the deployment supplied both a backend and its boundary. An
unconfigured service is left absent, so a call to it fails as an unreachable server and opens its
circuit rather than being mounted against a placeholder that returns nothing an agent would treat
as an answer. The analysis server likewise refuses to run anything when no isolated runtime is
configured.

### Reports and provenance

A job that reaches `completed` or `partial` with a report has published four artifacts, stored
under `tenants/<tenant>/jobs/<job id>/` in the object store:

| Artifact | Read it with |
|---|---|
| `report.json` | `GET /api/v1/jobs/{id}/report` |
| `report.md` | `GET /api/v1/jobs/{id}/report?format=markdown` |
| `provenance-manifest.json` | `GET /api/v1/jobs/{id}/manifest` |
| `evidence.json` | the object store (publishable evidence only) |

The manifest lists every finding, every evidence record (source, content hash, retrieval time,
classification, the tool call that produced it, and whether the source had drifted) and every
tool call the research made. It never carries excerpt text. `GET /api/v1/jobs/{id}/findings`
returns the claims with their verdicts at any point in the job, including while a reviewer is
deciding.

A job carries the clearance of the requester who created it, and its agents act under exactly
that clearance. Readers are held to their own: evidence, findings and audit records above a
reader's clearance are withheld (and counted in `X-Withheld-Count`), and a report or manifest that
draws on sources above it is refused with a 403.

Set `RESEARCH_ARTIFACT_ENDPOINT`, `RESEARCH_ARTIFACT_ACCESS_KEY`, `RESEARCH_ARTIFACT_SECRET_KEY`
and optionally `RESEARCH_ARTIFACT_BUCKET` to use MinIO or another S3-compatible store. Without
them artifacts stay in process memory, which an API process cannot read back from a worker.

### Budgets and usage

A job has four budgets: tool calls, tokens, estimated cost and working time. Tool calls are
claimed before each call. The other three are checked before each agent call, so a job that has
spent one is refused new work and ends `partial` with the reason (or `failed`, if it had found
nothing yet). Time spent waiting for a reviewer is not counted.

```json
{"question": "...", "budget": {"max_tool_calls": 50, "max_tokens": 2000000,
                               "max_cost_usd": 10.0, "max_runtime_seconds": 3600}}
```

`GET /api/v1/jobs/{id}` returns `usage`: what the job has spent so far. Cost is an estimate from
the provider's reported token counts and `RESEARCH_LLM_INPUT_COST_PER_MILLION_USD` /
`RESEARCH_LLM_OUTPUT_COST_PER_MILLION_USD`, which should match the model in `RESEARCH_AGENT_LLM`.

Set `RESEARCH_REDIS_URL` to count budgets and the web request rate limit in Redis. Every worker
and server replica then spends against one total, and each claim is atomic, so two workers can
never both take a job's last tool call. Without it each process counts its own, which is only
correct for a single worker. If Redis becomes unreachable the platform refuses the work it cannot
account for instead of running it unmetered.

### Evaluation

```powershell
python -m research_platform.evaluation          # scores and design targets
python -m research_platform.evaluation --json   # the full report
```

The suite runs the real pipeline over labelled scenarios against a fixed corpus, using the model
in `RESEARCH_AGENT_LLM`, and scores it on the section 14 criteria: task completion, tool selection,
schema-valid tool calls, citation correctness, claim support, research coverage, contradiction
recall, and cost and time per completed report. It also has a second tenant try every route to
another tenant's job and counts how many get through. The command exits non-zero when a design
target is missed. Recovery after a worker restart is proven separately, against a real Temporal
test server, in `tests/integration/test_workflow_recovery.py`.

## Known limits

- The evaluation suite has only been run with a scripted crew. Model quality scores need a run
  with a real model and its API key.
- The Python sandbox backend starts containers through a container runtime socket, which should
  not be mounted into a Kubernetes pod. Back it with a Job or a sandboxed runtime class there.
- Circuit breaker state is kept per worker, not shared.
- Agent collaboration is orchestrated by the Temporal workflow calling one CrewAI agent per step,
  rather than by a CrewAI Flow.
