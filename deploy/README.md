# Deploying Transformers CI Public Grafana

This directory packages the Transformers CI public observability stack for
Kubernetes. The Helm chart is intentionally self-contained and values-driven so
it can be deployed into a cluster without changing application code. Production
values (internal domains, certificate ARNs, secret wiring) are **not** committed
to this public repository — supply them through a private values file.

## Contents

- `helm/` contains the Helm chart. It deploys:
  - **OpenTelemetry Collector** — receives OTLP traces from CI jobs
  - **Grafana Tempo** — persistent trace storage (StatefulSet)
  - **pytest-trace-exporter** — custom Prometheus metrics exporter from traces
  - **Prometheus** — metrics storage (StatefulSet)
  - **Grafana** — public read-only dashboards
  - **CI data publisher** — optional scheduled publisher for trace-derived datasets (CronJob)
  - **NetworkPolicy** — optional cluster-boundary traffic policy for blast-radius reduction
- `helm/env/example.yaml` carries public-safe placeholder values. Copy it to a
  private file (e.g. `helm/env/private.yaml`, git-ignored) for a real deployment.
- `helm/transformers-ci-secrets.example.yaml` is a template for the sensitive
  runtime Secret. Copy it to `helm/transformers-ci-secrets.yaml` (git-ignored),
  fill it locally, and never commit it.
- `helm/dashboards/` holds the Grafana dashboard JSONs mounted into Grafana.
- `scripts/deploy.sh` checks the current Kubernetes context, creates the
  namespace when needed, optionally applies a local Secret file, runs Helm, and
  waits on each workload's rollout.
- `scripts/logs.sh` finds the current running pod for a given component and
  prints recent logs.

## Chart Behavior

Tempo and Prometheus run as StatefulSets backed by PVCs (`ebs-gp3` by default),
so trace and metric history survives pod restarts. Grafana, the OTel Collector,
and the trace exporter run as Deployments; the CI data publisher runs as a
CronJob.

Sensitive values are loaded from a pre-created Secret (referenced by
`secrets.name`); non-secret runtime config lives in `values.yaml` / your env
values file. Set `secrets.create: false` in production and provide the Secret
through your secret-management mechanism — either a local Secret file applied by
`deploy.sh --secret-file`, or the generic `externalSecret` block (a
provider-specific secret-sync CRD such as Infisical's `InfisicalSecret`).

The trace exporter and CI data publisher clone code from this repository at pod
startup. For code-only changes, pass a source revision so the pods pick it up
(see [Source Code Changes](#source-code-changes)).

## Deploy

Render manifests locally without touching the cluster:

```bash
deploy/scripts/deploy.sh --dry-run -f deploy/helm/env/private.yaml
```

Create or update the Secret in the target namespace and deploy:

```bash
cp deploy/helm/transformers-ci-secrets.example.yaml deploy/helm/transformers-ci-secrets.yaml
$EDITOR deploy/helm/transformers-ci-secrets.yaml
deploy/scripts/deploy.sh \
  -n transformers-ci \
  --secret-file deploy/helm/transformers-ci-secrets.yaml \
  -f deploy/helm/env/private.yaml
```

Deploy without applying a Secret file (Secret already exists, e.g. synced by an
external-secret operator):

```bash
deploy/scripts/deploy.sh -n transformers-ci -f deploy/helm/env/private.yaml
```

Use `--context` to refuse any other kube context:

```bash
deploy/scripts/deploy.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n transformers-ci \
  -f deploy/helm/env/private.yaml
```

## Logs

Fetch recent logs for a component (`grafana`, `otelcol`, `trace-exporter`,
`tempo`, `prometheus`, `ci-data-publisher`):

```bash
deploy/scripts/logs.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n transformers-ci \
  -c trace-exporter \
  --since 2h \
  --grep 'error|traceback|crashed|HTTPError'
```

Print only the latest error block:

```bash
deploy/scripts/logs.sh -n transformers-ci -c trace-exporter --since 2h --last-error
```

## Network Policies

Network policies are disabled by default so the example chart renders safely in
any cluster/CNI. To reduce blast radius from the public Grafana, trace-exporter,
and OTLP endpoints, enable the policy in private values and set `allowedCIDRs` to
the cluster/VPC address space that should be reachable:

```yaml
networkPolicy:
  enabled: true
  allowedCIDRs:
    - 10.0.0.0/8
  allowSameReleasePods: true
```

When enabled, the chart renders a `NetworkPolicy` selecting pods in this Helm
release with `policyTypes: [Ingress, Egress]`. It allows ingress from, and
egress to, only the configured CIDRs plus same-release pod-to-pod traffic.
Standard Kubernetes `NetworkPolicy` supports CIDRs, not DNS names — if the
deployment performs runtime source/package downloads, add tightly scoped
`networkPolicy.extraEgress` rules for the required address ranges.

## Secrets

The chart references a Secret by `secrets.name` (default
`transformers-ci-secrets`). The required key names are configurable under
`secrets.keys`:

| values key             | default Secret key       | used by           |
| ---------------------- | ------------------------ | ----------------- |
| `traceApiKey`          | `trace-api-key`          | OTLP ingestion    |
| `grafanaAdminPassword` | `grafana-admin-password` | Grafana admin     |
| `githubToken`          | `github-token`           | trace-exporter    |
| `hfToken`              | `hf-token`               | ci-data-publisher |

For production, set `secrets.create: false` and provide the Secret through your
preferred mechanism. The `externalSecret` block can have Helm manage a
provider-specific secret-sync CRD:

```yaml
secrets:
  create: false
  name: transformers-ci-secrets

externalSecret:
  enabled: true
  apiVersion: secrets.infisical.com/v1alpha1
  kind: InfisicalSecret
  spec:
    # Provider-specific configuration belongs in private values.
```

> The `githubToken` key matters for dashboard correctness: without it the
> trace-exporter calls the GitHub API unauthenticated (60 req/hr), gets
> rate-limited, and PR titles render blank in the dashboards.

## Keeping Docker Compose and Helm in Sync

The `dashboard/` directory is the **single source of truth** for the dashboard
JSONs (it's where contributors edit them, and the Docker Compose stack mounts
them directly). The entries under `deploy/helm/dashboards/` are **symlinks** into
`dashboard/` — Helm follows them when embedding the chart, so there is only ever
one real copy and nothing to sync. (Recreate a link with
`ln -s ../../../dashboard/<file> deploy/helm/dashboards/<file>`.)

The exporter helper endpoints use the same bare paths (`/failure`, `/badge`,
`/summary`) in both deployments, so a single dashboard JSON works for both.

### Dashboard Changes

Edit dashboards in `dashboard/` only — the chart symlinks pick the changes up
automatically.

Checklist:

- [ ] New dashboards: add the JSON file to `grafana.dashboards.files` in `helm/values.yaml`
- [ ] Version bump: run `dashboard/bump-version.sh X.Y.Z` (edits the canonical `dashboard/` copies)

### Source Code Changes

For changes to Helm templates or values, no manual restart is needed: Helm
updates the rendered manifests and Kubernetes rolls affected workloads when their
pod templates change.

The trace exporter and CI data publisher clone code from this repository at pod
startup. For code-only changes that do not otherwise change the rendered
manifests, pass a source revision during upgrade (a git commit SHA is
recommended):

```bash
helm upgrade transformers-ci ./deploy/helm \
  --namespace transformers-ci \
  -f deploy/helm/env/private.yaml \
  --set-string traceExporter.sourceRevision="$GIT_SHA" \
  --set-string ciDataPublisher.sourceRevision="$GIT_SHA"
```

These values render as pod template annotations, so changing them triggers a
rollout for `trace-exporter` and updates the `ci-data-publisher` CronJob pod
template for future jobs.

### Deployment Verification

`deploy.sh` waits on every workload, but to check manually:

```bash
kubectl rollout status deployment/grafana -n transformers-ci
kubectl rollout status deployment/otelcol -n transformers-ci
kubectl rollout status deployment/trace-exporter -n transformers-ci
kubectl rollout status statefulset/tempo -n transformers-ci
kubectl rollout status statefulset/prometheus -n transformers-ci
```
