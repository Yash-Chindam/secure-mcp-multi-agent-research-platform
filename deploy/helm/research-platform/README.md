# research-platform

The section 15 deployment topology, as a Helm chart. The application and MCP planes
(`api`, `worker`, the section 8 servers once they run standalone) are this chart's own
templates; the data and observability planes are composed from maintained upstream
charts declared in `Chart.yaml` - see its comment for why, and `values.yaml`'s comments
for what had to be set on each to get a working default (a shared SQL store for
Temporal, filesystem-backed Loki, an OTLP pipeline into Tempo and Prometheus for the
otel-collector).

## Install

```bash
helm dependency update deploy/helm/research-platform
helm install research-platform deploy/helm/research-platform \
  --set-file opa.policyRego=policy/research/authz.rego
```

`helm lint` and `helm template` are both clean against the versions `Chart.lock` pins;
re-run `helm dependency update` if you bump a version, and re-verify both before
assuming a new one still renders (a later chart version can, and did during this
chart's own development, change a required value's default).

## What isn't wired up automatically

- **Keycloak**: no realm or client is provisioned. Create one and set
  `settings.oidcIssuer`/`settings.oidcAudience` (or `existingSecretName` if you'd rather
  keep the audience out of a ConfigMap) before the platform verifies tokens rather than
  falling back to development identity headers.
- **OPA**: `opa.policyRego` is empty by default - the `--set-file` above is required, not
  optional, for OPA to do anything but refuse every request.
- **MCP servers**: `mcpServers` is empty by default. Every section 8 server runs
  in-process inside `api`/`worker` today (see `research_platform/worker.py`'s
  docstring), so there is nothing to list here yet.
- **A restricted database role**: `settings.databaseUrl` connects the api and worker to
  the bundled PostgreSQL as its owner, which works but bypasses nothing only because
  the schema forces row-level security on the owner too. A superuser, or any role with
  BYPASSRLS, is exempt from those policies entirely - the platform logs a warning at
  startup when it finds itself connected as one. For production, create a dedicated
  non-superuser role with SELECT/INSERT/UPDATE on the three tables and supply its URL
  as `RESEARCH_DATABASE_URL` through `existingSecretName`, leaving
  `settings.databaseUrl` empty.
