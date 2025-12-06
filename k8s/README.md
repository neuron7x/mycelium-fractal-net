# Kubernetes Deployment for MyceliumFractalNet v4.1

This directory contains Kubernetes manifests for deploying MyceliumFractalNet in production.

## Directory Structure

```
k8s/
├── README.md           # This file
├── base/               # Base Kustomize configuration
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── secrets.yaml    # Secrets template (replace placeholders!)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── network-policy.yaml
│   └── ingress.yaml
└── k8s.yaml (DEPRECATED - use base/ instead)
```

## Quick Start

### 1. Configure Secrets

**IMPORTANT**: Never commit actual secrets to version control!

```bash
# Generate a secure API key
python -c 'import secrets; print(secrets.token_urlsafe(32))'

# Edit secrets template
vim k8s/base/secrets.yaml

# Replace REPLACE_WITH_ACTUAL_KEY with your generated key
```

### 2. Configure Ingress

Edit `k8s/base/ingress.yaml` and replace `mfn.example.com` with your actual domain.

### 3. Deploy to Kubernetes

```bash
# Apply all manifests using Kustomize
kubectl apply -k k8s/base/

# Or apply individually
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/secrets.yaml
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f k8s/base/hpa.yaml
kubectl apply -f k8s/base/pdb.yaml
kubectl apply -f k8s/base/network-policy.yaml
kubectl apply -f k8s/base/ingress.yaml
```

### 4. Verify Deployment

```bash
# Check all resources
kubectl get all -n mycelium-fractal-net

# Check pods are running
kubectl get pods -n mycelium-fractal-net

# Check service is exposed
kubectl get svc -n mycelium-fractal-net

# Check ingress is configured
kubectl get ingress -n mycelium-fractal-net

# View logs
kubectl logs -n mycelium-fractal-net -l app=mycelium-fractal-net --tail=100
```

## Production Features

### High Availability

- **Replicas**: 3 pods by default
- **PodDisruptionBudget**: Ensures minimum 2 pods during updates
- **HorizontalPodAutoscaler**: Auto-scales 1-10 pods based on CPU/memory

### Security

- **NetworkPolicy**: Restricts ingress/egress traffic
- **Secrets**: API keys stored securely
- **TLS**: Ingress configured for HTTPS (requires cert-manager)
- **Non-root**: Container runs as non-root user

### Resource Management

**Requests** (guaranteed):
- Memory: 256Mi
- CPU: 250m (0.25 cores)

**Limits** (maximum):
- Memory: 512Mi
- CPU: 500m (0.5 cores)

### Health Checks

- **Liveness Probe**: /health endpoint (port 8000)
- **Readiness Probe**: /health endpoint (port 8000)

## Secrets Management

### Development/Staging

For non-production environments, you can use the secrets template directly:

```bash
kubectl create secret generic mfn-secrets \
  --from-literal=api-key=your-dev-key \
  -n mycelium-fractal-net
```

### Production

Use external secrets management for production:

**Option 1: External Secrets Operator**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mfn-secrets
  namespace: mycelium-fractal-net
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: mfn-secrets
  data:
    - secretKey: api-key
      remoteRef:
        key: mfn/api-key
```

**Option 2: Sealed Secrets**
```bash
# Install kubeseal
# Encrypt secrets
kubeseal -f k8s/base/secrets.yaml -o yaml > k8s/base/sealed-secrets.yaml
# Commit sealed-secrets.yaml (safe to commit)
```

**Option 3: Cloud Provider Secrets**
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` endpoint:
- `mfn_http_requests_total`
- `mfn_http_request_duration_seconds`
- `mfn_http_requests_in_progress`

### ServiceMonitor (if using Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mycelium-fractal-net
  namespace: mycelium-fractal-net
spec:
  selector:
    matchLabels:
      app: mycelium-fractal-net
  endpoints:
    - port: http
      path: /metrics
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment mycelium-fractal-net \
  --replicas=5 \
  -n mycelium-fractal-net
```

### Auto-scaling (HPA)

HPA is configured to scale 1-10 pods based on:
- CPU: >50%
- Memory: >80%

Adjust in `k8s/base/hpa.yaml`.

## Troubleshooting

### Pods not starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n mycelium-fractal-net

# Check logs
kubectl logs <pod-name> -n mycelium-fractal-net
```

### Service not accessible

```bash
# Check service endpoints
kubectl get endpoints -n mycelium-fractal-net

# Check ingress
kubectl describe ingress mycelium-fractal-net-ingress -n mycelium-fractal-net
```

### Secrets not found

```bash
# Verify secrets exist
kubectl get secrets -n mycelium-fractal-net

# Check secret content (base64 encoded)
kubectl get secret mfn-secrets -n mycelium-fractal-net -o yaml
```

## Updates and Rollbacks

### Rolling Update

```bash
# Update image
kubectl set image deployment/mycelium-fractal-net \
  mycelium-fractal-net=mycelium-fractal-net:v4.2 \
  -n mycelium-fractal-net

# Watch rollout
kubectl rollout status deployment/mycelium-fractal-net \
  -n mycelium-fractal-net
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/mycelium-fractal-net \
  -n mycelium-fractal-net

# Rollback to previous version
kubectl rollout undo deployment/mycelium-fractal-net \
  -n mycelium-fractal-net
```

## Security Hardening

### Network Policies

NetworkPolicy is configured to:
- Allow ingress only from ingress controller
- Allow DNS resolution (UDP 53)
- Allow HTTPS egress (TCP 443)
- Block all other traffic by default

### Pod Security

Deployment is configured with:
- `runAsNonRoot: true`
- `allowPrivilegeEscalation: false`
- `readOnlyRootFilesystem: true` (if applicable)

### Secrets Rotation

Regular secret rotation is recommended:

```bash
# Generate new API key
NEW_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')

# Update secret
kubectl create secret generic mfn-secrets \
  --from-literal=api-key=$NEW_KEY \
  -n mycelium-fractal-net \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secret
kubectl rollout restart deployment/mycelium-fractal-net \
  -n mycelium-fractal-net
```

## Migration from k8s.yaml

The legacy `k8s.yaml` file has been split into modular files in `k8s/base/`.

To migrate:

1. Delete old deployment: `kubectl delete -f k8s.yaml`
2. Apply new manifests: `kubectl apply -k k8s/base/`
3. Verify deployment: `kubectl get all -n mycelium-fractal-net`

**Note**: The legacy `k8s.yaml` is kept for backward compatibility but is deprecated.

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize](https://kustomize.io/)
- [External Secrets Operator](https://external-secrets.io/)
- [Technical Audit](../docs/TECH_DEBT_AUDIT_2025.md)

---

**Last Updated**: 2025-12-06  
**Version**: 4.1.0
