# Secure MCP Multi-Agent Research Platform

**Document type:** Technical design specification  
**Purpose:** Define a production-grade, governed multi-agent research system  
**Implementation plan:** Intentionally excluded

> Proposed quality and performance values are design targets. Use them on a resume only after measuring a working implementation.

## 1. Use case

Organizations need research reports that combine public web information, internal documents, structured databases, GitHub repositories and reproducible calculations. Typical agent demos hard-code their tools, lose state on failure, offer weak source provenance and give agents excessive permissions.

This platform accepts a research assignment, decomposes it into tasks, coordinates specialist agents through CrewAI, dynamically discovers authorized tools through MCP, records source-level evidence, pauses for human approval when required and produces a cited report with a complete execution trail.

### Representative scenarios

1. Compare AI vendors, pricing, benchmarks and release claims using dated evidence.
2. Analyze public financial reports with approved internal datasets without exposing trading actions.
3. Research regulations while preserving jurisdiction, effective date and citation provenance.
4. Inspect approved code repositories and run isolated analytical scripts.

## 2. Portfolio value

The project demonstrates:

- Model Context Protocol architecture.
- CrewAI multi-agent collaboration.
- Dynamic tool discovery.
- Durable long-running workflows.
- Human-in-the-loop approvals.
- OAuth identity and policy-as-code authorization.
- Evidence provenance and citation evaluation.
- Distributed observability and failure recovery.

## 3. Users and actors

| Actor | Responsibility |
|---|---|
| Research requester | Submits the question, constraints, source requirements and report format. |
| Reviewer | Approves scope, sensitive actions and final publication. |
| Platform administrator | Registers MCP servers, policies, tenants and limits. |
| Planner agent | Decomposes the assignment and specifies evidence requirements. |
| Researcher agents | Collect evidence through approved MCP tools. |
| Analyst agent | Compares evidence and performs calculations. |
| Critic agent | Identifies unsupported claims, conflicts and coverage gaps. |
| Reporter agent | Produces the report using approved findings and citation IDs. |
| MCP providers | Expose independent tool, prompt and resource services. |

## 4. Scope

### In scope

- Multi-source, long-running research assignments.
- CrewAI specialist agents and CrewAI Flows.
- Dynamic MCP capability discovery.
- Web, filesystem, PostgreSQL, GitHub and Python-analysis MCP servers.
- Parallel evidence gathering.
- Claim-to-source provenance.
- Durable job state and recovery.
- Human approval checkpoints.
- Tenant- and role-aware tool authorization.
- Quality, latency, cost and security evaluation.

### Out of scope

- Unrestricted autonomous publication.
- External side effects without explicit approval.
- General browser control outside approved research tools.
- Model training or fine-tuning.
- Bypassing source restrictions or access controls.
- Accepting model-generated citations without evidence verification.

## 5. Architecture

```text
Research UI / FastAPI
        |
        v
Keycloak authentication and tenant context
        |
        v
CrewAI Flow inside a Temporal workflow
        |
        +---------------- Human approval signals ----------------+
        |                                                         |
        v                                                         |
CrewAI Crew                                                      |
Planner -> Researchers -> Analyst -> Critic -> Reporter           |
        |                                                         |
        v                                                         |
MCP capability registry + OPA policy interceptor <---------------+
        |
        +---- Web Research MCP server
        +---- Filesystem MCP server
        +---- PostgreSQL MCP server
        +---- GitHub MCP server
        +---- Sandboxed Python MCP server
        |
        v
PostgreSQL + Redis + MinIO
        |
        v
OpenTelemetry -> Prometheus / Grafana / Loki / Tempo
```

## 6. Technology selection

| Technology | Responsibility | Selection rationale |
|---|---|---|
| CrewAI | Role-based agent collaboration | Natural fit for planner, researcher, analyst, critic and reporter roles. |
| CrewAI Flows | Deterministic lifecycle | Adds explicit states and branching around autonomous crew execution. |
| FastMCP | MCP clients and servers | Supports remote tools, resources, prompts and authenticated HTTP deployment. |
| Temporal | Durable execution | Preserves long-running jobs, retries, timers and human-wait states. |
| FastAPI | Application API | Typed asynchronous APIs with OpenAPI support. |
| Keycloak | Authentication | Provides OIDC/OAuth identity for users and services. |
| Open Policy Agent | Authorization | Keeps tool and resource permissions explicit and testable. |
| PostgreSQL | System of record | Stores jobs, evidence metadata, findings, approvals and audit references. |
| Redis | Short-lived state | Supports cache, locks, limits and selected coordination state. |
| MinIO | Artifact storage | Stores evidence bundles, calculations and generated reports. |
| OpenTelemetry | Distributed tracing | Correlates workflows, agent tasks, MCP calls and downstream services. |

## 7. Agent model

| Agent | Input | Output | Tool access |
|---|---|---|---|
| Planner | Research request | Structured tasks and evidence requirements | Tool metadata only |
| Researcher | One task and source policy | `EvidenceRecord[]` | Approved research tools |
| Analyst | Evidence records | Findings and calculations | Read-only data and Python tools |
| Critic | Findings and evidence graph | Verdicts and revision issues | Evidence retrieval only |
| Reporter | Approved findings | Cited report sections | No side-effecting tools |

All agent outputs use Pydantic schemas. An invalid response cannot advance the workflow state.

## 8. MCP service design

| MCP server | Capabilities | Required boundary |
|---|---|---|
| Web research | Search, fetch and extract approved public sources | Domain policy, rate limit and response-size limit |
| Filesystem | List and read approved workspace content | Tenant-scoped roots and read-only default |
| PostgreSQL | Inspect schemas and execute analytical queries | Read-only role, query parser and row-level security |
| GitHub | Read repositories, issues and pull-request metadata | Repository allowlist and scoped OAuth token |
| Python analysis | Perform calculations and generate artifacts | Ephemeral container, disabled network and resource limits |

Remote services use Streamable HTTP. Capability discovery reveals only tools visible to the current identity. Authorization is checked again immediately before execution.

## 9. Core workflow

```text
Authenticate requester
    -> create ResearchJob
    -> plan tasks and evidence requirements
    -> discover authorized MCP capabilities
    -> collect evidence in parallel
    -> normalize, classify and hash evidence
    -> analyze findings and calculations
    -> critic verifies claims and coverage
    -> reviewer resolves flagged issues
    -> reporter creates the final report
    -> export report and provenance manifest
```

Every factual claim must reference evidence identifiers. The reporter cannot introduce uncited factual content.

## 10. Information model

### ResearchJob

- Job and tenant IDs.
- Requester identity.
- Question, constraints and source requirements.
- Token, monetary, tool-call and time budgets.
- Status and durable workflow checkpoint.

### ResearchTask

- Objective and assigned agent.
- Dependencies.
- Evidence requirements.
- Source restrictions.
- Retry count and state.

### EvidenceRecord

- Normalized excerpt.
- Source URL or resource ID.
- Title, author and publication date where available.
- Retrieval time and content hash.
- Trust and access classifications.
- Producing task and tool invocation.

### Finding

- Claim.
- Supporting and contradicting evidence IDs.
- Calculation IDs.
- Confidence and critic verdict.
- Reviewer status.

### ToolInvocation

- MCP server and capability.
- Sanitized arguments.
- Policy version and authorization decision.
- Timing, outcome and error classification.

## 11. Security design

- Use short-lived OAuth tokens instead of shared long-lived keys.
- Enforce isolation at database, artifact and tool boundaries.
- Treat retrieved content as untrusted evidence, never system instruction.
- Keep source text separate from tool-control messages.
- Require approval for side effects and sensitive resources.
- Bind approval to the exact tool, resource and argument digest.
- Store secrets outside code and exclude them from prompts and traces.
- Sanitize tool results before returning them to agent context.
- Limit recursion, execution time, result size and tool-call count.

## 12. Reliability design

| Failure | Required behavior |
|---|---|
| MCP server unavailable | Open a circuit and use an alternative only when policy and semantics allow it. |
| Invalid agent output | Reject the state transition and request bounded schema correction. |
| Source content changed | Preserve the original evidence and flag content drift. |
| Approval delayed | Suspend durably without consuming an active worker. |
| Budget exhausted | Stop new work and return clearly labeled partial findings. |
| Worker restarted | Resume from Temporal history without duplicating completed side effects. |

## 13. Observability

```text
Research job
  -> Temporal workflow/activity
    -> CrewAI Flow stage
      -> agent task
        -> MCP invocation
          -> downstream HTTP/database action
```

Track:

- Queue age and active jobs.
- Task completion and failure rate.
- MCP latency, error rate and circuit state.
- Retry counts.
- Approval wait time.
- Tokens and estimated cost per job.
- Citation correctness and claim support.
- Permission denials and injection flags.

## 14. Evaluation

- Tool-selection accuracy.
- Task-completion rate.
- Citation correctness.
- Claim-support rate.
- Research coverage.
- Contradiction-identification recall.
- Recovery after controlled failures.
- Cross-tenant access prevention.
- Cost and time per completed report.

### Proposed design targets

- At least 95% schema-valid tool calls on the evaluation suite.
- Every published factual claim linked to evidence.
- Zero successful cross-tenant access attempts.
- Workflow resumes after restart without repeating completed actions.
- Partial output clearly identified when full completion is impossible.

## 15. Deployment topology

- FastAPI ingress deployment.
- Temporal server or managed Temporal service.
- Separate Temporal/CrewAI workers.
- Independently scalable FastMCP services.
- PostgreSQL, Redis and MinIO.
- Keycloak and OPA.
- OpenTelemetry Collector.
- Prometheus, Grafana, Loki and Tempo.
- Docker, Kubernetes, Helm and GitHub Actions.
- NetworkPolicies separating application, MCP, data and observability planes.

## 16. Official references

- [Model Context Protocol architecture](https://modelcontextprotocol.io/specification/2025-06-18/architecture)
- [FastMCP HTTP deployment](https://gofastmcp.com/v2/deployment/http)
- [FastMCP authentication](https://gofastmcp.com/v2/servers/auth/authentication)
- [CrewAI documentation](https://docs.crewai.com/)
- [Temporal documentation](https://docs.temporal.io/)
- [Open Policy Agent documentation](https://www.openpolicyagent.org/docs/)

