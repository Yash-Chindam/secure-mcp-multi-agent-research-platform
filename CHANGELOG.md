# Changelog

## 0.9.0

The first tagged release. It implements the design specification in
[`01-secure-mcp-multi-agent-research-platform.md`](./01-secure-mcp-multi-agent-research-platform.md),
with the gaps listed under "Known limits" below.

### The pipeline (section 9)

- A job submitted through the API starts a durable Temporal workflow: plan, collect evidence in
  parallel, analyze, critique, pause for a reviewer when the critic flags something, report, and
  export.
- Five CrewAI agents (planner, researcher, analyst, critic, reporter), each bound to a Pydantic
  output contract with bounded correction.
- A reviewer decides through `POST /jobs/{id}/review`. The workflow waits durably, without holding
  a worker.

### Evidence and provenance (sections 4, 9, 12)

- The platform verifies evidence itself. An excerpt must appear in what a recorded tool call
  returned; the source, hash and classification come from that call, not from the agent.
- Every later citation must name recorded evidence, and a report may not cite restricted evidence.
- A finished job publishes its report (JSON and Markdown), a provenance manifest and an evidence
  bundle to MinIO, under a per-tenant prefix.
- Cited sources are re-read at publication. A source whose captured excerpt is gone is flagged as
  drifted, and the original evidence is kept.

### MCP services (section 8)

- Web research, filesystem, PostgreSQL, GitHub, Python sandbox and evidence servers on FastMCP,
  with production backends.
- Any server can run as its own deployment over Streamable HTTP, called with a short-lived service
  token.

### Security (section 11)

- OAuth access tokens verified against Keycloak; authorization in OPA, layered with the registry
  boundary and failing closed.
- Tenant isolation at the database (forced row-level security), the artifact store (per-tenant
  prefix) and every tool boundary.
- A job carries its requester's clearance and its agents act under it. Readers see only the
  evidence, findings, audit records and reports their own clearance covers.
- Approvals bound to the exact tool, resource and argument digest. Tool results sanitized and
  flagged for injection before they reach an agent.

### Reliability (section 12)

- Token, cost, tool-call and working-time budgets. A job that spends one ends partial and says
  which.
- Budgets and the web rate limit are counted in Redis, so every worker shares them. They fail
  closed if Redis is unreachable.
- Per-server circuit breaking.
- A job resumes on a different worker after a restart without repeating completed steps.
- A failed job is marked failed with the reason, rather than left in a working status.

### Observability and evaluation (sections 13, 14)

- OpenTelemetry traces and metrics: queue age, active jobs, task outcomes, MCP latency and errors,
  circuit state, retries, reviewer wait, tokens and cost per job, claim support, denials and
  injection flags.
- `python -m research_platform.evaluation` runs labelled scenarios through the real pipeline,
  scores every section 14 criterion and checks the design targets, including cross-tenant probes.

### Deployment (section 15)

- Docker Compose for local development; a Helm chart with separate API, worker and MCP server
  deployments, NetworkPolicies between planes, and PostgreSQL, Redis, MinIO, Keycloak, OPA,
  Temporal and the observability stack as dependencies.

### Known limits

- The pipeline has been exercised end to end with a scripted crew, not yet with a real model. The
  evaluation suite reports model quality only when run with one.
- The Python sandbox backend needs a container runtime socket, which should not be mounted into a
  Kubernetes pod.
- Circuit breaker state is per worker.
- Agents are orchestrated by the Temporal workflow calling one CrewAI agent per step, not by a
  CrewAI Flow.
- The Docker Compose stack runs without token verification and connects to PostgreSQL as a role
  that bypasses row-level security. Both are logged as warnings at startup.
