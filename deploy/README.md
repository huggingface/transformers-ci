# Transformers CI Public Grafana - Helm Chart

This directory contains the Helm chart for deploying the Transformers CI public-facing Grafana instance and supporting observability infrastructure.

## Overview

This chart deploys:

- **OpenTelemetry Collector** - Receives OTLP traces from CI jobs
- **Grafana Tempo** - Persistent trace storage
- **pytest-trace-exporter** - Custom metrics exporter from traces
- **Prometheus** - Metrics storage
- **Grafana** - Public read-only dashboards
- **CI data publisher** - Optional scheduled publisher for trace-derived datasets
- **NetworkPolicy** - Optional cluster-boundary traffic policy for blast-radius reduction

## Deployment

Production deployment values are intentionally not committed to this public repository. Use a private values file that provides deployment-specific domains, certificate ARNs, and secret-management configuration.

Example render/install command using the placeholder example values:

```bash
helm upgrade --install transformers-ci . \
  --namespace transformers-ci \
  --create-namespace \
  -f env/example.yaml
```

For a real deployment, replace `env/example.yaml` with a private values file.

## Network Policies

Network policies are disabled by default so the public example chart renders safely in any cluster/CNI. To reduce blast radius from the public Grafana, trace-exporter, and OTLP endpoints, enable the chart policy in private values and set `allowedCIDRs` to the cluster/VPC address space that should be reachable:

```yaml
networkPolicy:
  enabled: true
  allowedCIDRs:
    - 10.0.0.0/8
  allowSameReleasePods: true
```

When enabled, the chart renders a `NetworkPolicy` selecting pods in this Helm release with `policyTypes: [Ingress, Egress]`. It allows ingress from, and egress to, only the configured CIDRs plus same-release pod-to-pod traffic. This is intended to prevent a compromised public-facing pod from using arbitrary outbound network access to reach another cluster or public network.

If this deployment still performs runtime source/package downloads or publishing, remember that standard Kubernetes `NetworkPolicy` supports CIDRs, not DNS names. Either avoid those runtime internet dependencies in production or add tightly scoped `networkPolicy.extraEgress` rules for the required address ranges.

## Secrets

By default, the chart can create a placeholder Kubernetes Secret for local/manual testing. For production deployments, set:

```yaml
secrets:
  create: false
  name: transformers-ci-secrets
```

Then provide that Secret through your preferred private secret-management mechanism. The required Secret key names are configurable under `secrets.keys`; the committed defaults are placeholders.

Alternatively, private deployment values can let Helm manage a provider-specific secret-sync CRD through the generic `externalSecret` block:

```yaml
secrets:
  create: false
  name: transformers-ci-secrets

externalSecret:
  enabled: true
  apiVersion: example.com/v1
  kind: ExampleSecretSync
  spec:
    # Provider-specific configuration belongs in private values.
    # Values can use Helm templating via tpl.
```

## File Structure

```text
deploy/
├── Chart.yaml                    # Helm chart metadata
├── values.yaml                   # Default values with public-safe placeholders
├── env/
│   └── example.yaml              # Example override values with placeholders
├── templates/                    # Kubernetes manifests
│   ├── _helpers.tpl
│   ├── namespace.yaml
│   ├── otelcol.yaml              # OTel Collector
│   ├── tempo.yaml                # Tempo StatefulSet
│   ├── trace-exporter.yaml       # Custom metrics exporter
│   ├── prometheus.yaml           # Prometheus StatefulSet
│   ├── grafana.yaml              # Grafana Deployment and dashboards
│   ├── ingress.yaml              # Ingresses
│   ├── ci-data-publisher.yaml    # Optional publishing CronJob
│   ├── networkpolicy.yaml        # Optional cluster-boundary NetworkPolicy
│   └── secrets.yaml              # Generic Secret placeholder / external secret resource
└── dashboards/                   # Grafana dashboard JSONs
```

## Keeping Docker Compose and Helm in Sync

The `dashboard/` directory contains the Docker Compose configuration. The `deploy/` directory is the Kubernetes/Helm source of truth. Both live in this repo, so dashboard and observability changes can be made together.

### Dashboard Changes

When dashboards are updated in `dashboard/`:

```bash
cp dashboard/*.json deploy/dashboards/
```

Checklist:

- [ ] New dashboards: add the JSON file to `grafana.dashboards.files` in `values.yaml`
- [ ] Existing dashboards: copy the updated JSON into `deploy/dashboards/`

### Source Code Changes

For changes to Helm templates or values, no manual restart is needed: Helm updates the rendered manifests and Kubernetes rolls affected workloads when their pod templates change.

The trace exporter and CI data publisher also clone code from this repository at pod startup. For code-only changes that do not otherwise change the rendered manifests, pass a source revision value during upgrade. A git commit SHA is recommended:

```bash
helm upgrade transformers-ci ./deploy \
  --namespace transformers-ci \
  -f /path/to/private-values.yaml \
  --set-string traceExporter.sourceRevision="$GIT_SHA" \
  --set-string ciDataPublisher.sourceRevision="$GIT_SHA"
```

These values are rendered as pod template annotations, so changing them triggers a Kubernetes rollout for `trace-exporter` and updates the `ci-data-publisher` CronJob pod template for future jobs.

For new environment variables, add them to the relevant template under `templates/` and `values.yaml`; no separate rollout command is required.

### Deployment Verification

```bash
kubectl rollout status deployment/grafana -n transformers-ci
kubectl rollout status deployment/trace-exporter -n transformers-ci
kubectl rollout status statefulset/tempo -n transformers-ci
kubectl rollout status statefulset/prometheus -n transformers-ci
```
