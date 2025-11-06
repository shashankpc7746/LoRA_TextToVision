# 🔒 BHIV Security CI/CD Integration Guide

## Overview

This document explains the automated security workflows integrated into the CI/CD pipeline for the BHIV (Bharatiya Hyperscale Infrastructure Vision) project.

## Workflows

### 1. Security Artifact Signing (`security-artifact-signing.yml`)

**Purpose**: Automatically signs all training artifacts (models, checkpoints, adapters) with Ed25519 cryptographic signatures.

**Trigger**:
- After successful training pipeline completion
- Manual workflow dispatch for specific artifacts
- On push to main/task_quality_leap branches

**What it does**:
1. Generates Ed25519 signing key pair (in production, use GitHub Secrets)
2. Searches for unsigned artifacts in:
   - `adapters/gurukul_lora/**/*.{safetensors,ckpt,pth,bin}`
   - `AnimateDiff/models/**/*.{safetensors,ckpt,pth,bin}`
3. Signs each artifact with Ed25519 signature
4. Creates `.sig` sidecar files for each artifact
5. Verifies all signatures
6. Commits signatures back to repository
7. Uploads public key as artifact
8. Generates security report

**Security Gate**: Fails if unsigned artifacts found in production paths

**Manual Usage**:
```bash
# Trigger workflow for specific artifact
gh workflow run security-artifact-signing.yml \
  -f artifact_path="adapters/gurukul_lora/my-model.safetensors"
```

---

### 2. Docker Image Signing (`security-docker-signing.yml`)

**Purpose**: Signs Docker container images using Cosign (Sigstore) for supply chain security.

**Trigger**:
- After Docker build completion
- On Dockerfile changes
- Manual workflow dispatch

**What it does**:
1. Installs Cosign (container signing tool)
2. Builds Docker image with metadata
3. Pushes to GitHub Container Registry (ghcr.io)
4. Signs image with Cosign (keyless OIDC)
5. Generates SBOM (Software Bill of Materials)
6. Signs SBOM and attaches to image
7. Runs Trivy security scan for vulnerabilities
8. Verifies signature before deployment approval
9. Blocks deployment if signature invalid

**Security Features**:
- **Keyless Signing**: Uses GitHub OIDC (no key management)
- **SBOM**: Tracks all dependencies
- **Vulnerability Scan**: Fails on CRITICAL/HIGH vulnerabilities
- **Deployment Gate**: Only signed images can deploy

**Verification**:
```bash
# Verify signed image
cosign verify ghcr.io/shashankpc7746/lora_texttovision:latest \
  --certificate-identity-regexp=".*" \
  --certificate-oidc-issuer-regexp=".*"
```

---

### 3. Security Gates (`security-gates.yml`)

**Purpose**: Comprehensive security validation on every PR and push.

**Trigger**:
- Pull requests to main/task_quality_leap
- Push to main/task_quality_leap
- Manual workflow dispatch

**Security Checks**:

#### 3.1 Security Linting
- **Bandit**: Python security vulnerability scanner
- **Safety**: Checks dependencies for known CVEs
- **Semgrep**: Static analysis for security bugs
- **Secret Detection**: Scans for hardcoded passwords/tokens

#### 3.2 Artifact Signature Verification
- Checks signature coverage (must be 100%)
- Verifies signature validity
- Fails if coverage < 80%

#### 3.3 Watermarking Validation
- Verifies invisible watermarking module
- Verifies visible watermarking module
- Checks BHI logo presence

#### 3.4 Encryption Validation
- Tests KSML encryption (AES-256-GCM)
- Tests encryption/decryption roundtrip
- Validates key derivation (PBKDF2HMAC)

#### 3.5 Runtime Key Validation
- Tests key issuance system
- Tests key validation system
- Verifies time-limited keys

**Final Gate**: All checks must pass for PR merge

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BHIV Security Pipeline                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ Encrypt │    │  Sign   │    │Watermark│
         │ (KSML)  │    │(Ed25519)│    │(Multi)  │
         └────┬────┘    └────┬────┘    └────┬────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                         ┌────▼────┐
                         │ Audit   │
                         │  Log    │
                         └────┬────┘
                              │
                    ┌─────────▼─────────┐
                    │   Runtime Auth    │
                    │ (Time-limited)    │
                    └─────────┬─────────┘
                              │
                         ┌────▼────┐
                         │Deployment│
                         └─────────┘
```

---

## Key Management

### Artifact Signing Keys (Ed25519)

**Current Setup** (for CI/CD):
- Keys generated in workflow
- Public key saved to `security/keys/signing_key.pub`
- Private key ephemeral (not persisted)

**Production Setup** (recommended):
```yaml
# In GitHub Secrets:
ARTIFACT_SIGNING_PRIVATE_KEY: <Ed25519 private key PEM>
ARTIFACT_SIGNING_PUBLIC_KEY: <Ed25519 public key PEM>

# Workflow usage:
- name: Load signing key
  env:
    SIGNING_KEY: ${{ secrets.ARTIFACT_SIGNING_PRIVATE_KEY }}
  run: |
    echo "$SIGNING_KEY" > security/keys/signing_key.pem
```

**Key Rotation**:
- Rotate every 90 days
- Keep previous public key for 180 days (verification)
- Update all workers with new public key

### Docker Image Signing (Cosign)

**Keyless Mode** (current):
- Uses GitHub OIDC for identity
- No key management required
- Signatures stored in Rekor transparency log

**Key-based Mode** (optional):
```bash
# Generate cosign keys
cosign generate-key-pair

# Store in GitHub Secrets:
# COSIGN_PRIVATE_KEY
# COSIGN_PASSWORD
```

### KSML Encryption

**Production Setup**:
```bash
# Generate strong KSML token
openssl rand -hex 32 > ksml_token.txt

# Store in GitHub Secrets:
# KSML_TOKEN: <generated token>
```

---

## Monitoring & Alerts

### Security Metrics

**Artifact Signing Coverage**:
```bash
# Check coverage
python << 'EOF'
from pathlib import Path
artifacts = list(Path('.').rglob('*.safetensors'))
signed = [a for a in artifacts if Path(str(a) + '.sig').exists()]
print(f"Coverage: {len(signed)/len(artifacts)*100:.1f}%")
EOF
```

**Docker Image Verification**:
```bash
# Verify latest image
cosign verify ghcr.io/shashankpc7746/lora_texttovision:latest
```

### Alerts

**GitHub Actions Notifications**:
- Workflow failures → Email/Slack
- Security scan findings → GitHub Security tab
- Unsigned artifacts → PR comment

---

## Security Best Practices

### 1. Never Commit Secrets
```python
# ❌ BAD
api_key = "sk-1234567890abcdef"

# ✅ GOOD
import os
api_key = os.environ.get('API_KEY')
```

### 2. Always Verify Signatures
```python
from security.artifact_signer import ArtifactSigner

signer = ArtifactSigner('security/keys/signing_key.pub')
if not signer.verify_signature('model.safetensors'):
    raise SecurityError("Invalid signature!")
```

### 3. Use Time-Limited Keys
```python
from security.runtime_validator import RuntimeKeyIssuer
from datetime import timedelta

issuer = RuntimeKeyIssuer()
key = issuer.issue_runtime_key(
    worker_id="worker-001",
    ttl=timedelta(hours=12)  # 12-hour validity
)
```

### 4. Enable All Watermarks
```python
# Invisible watermark (metadata + LSB)
from security.watermark import VideoWatermarker
watermarker = VideoWatermarker()
watermarker.embed_metadata_watermark(video, metadata)

# Visible watermark (logo)
from security.visible_watermark import VisibleWatermarker
visible = VisibleWatermarker()
visible.add_corner_watermark(video, opacity=0.15)
```

---

## Troubleshooting

### Signature Verification Fails

**Problem**: `verify_signature()` returns False

**Solutions**:
1. Check public key matches signing key:
   ```bash
   openssl pkey -in signing_key.pem -pubout
   ```
2. Re-sign artifact:
   ```bash
   python -c "from security.artifact_signer import ArtifactSigner; \
              ArtifactSigner('signing_key.pem').sign_artifact('model.safetensors')"
   ```
3. Verify file integrity (not corrupted)

### Docker Image Signing Fails

**Problem**: Cosign verification fails

**Solutions**:
1. Check COSIGN_EXPERIMENTAL=1 is set
2. Verify image digest:
   ```bash
   docker inspect ghcr.io/user/image:tag | grep Digest
   ```
3. Check Rekor transparency log:
   ```bash
   rekor-cli search --artifact <image>
   ```

### Security Gate Failures

**Problem**: CI security checks fail

**Solutions**:
1. Review workflow logs in Actions tab
2. Run locally:
   ```bash
   bandit -r . -ll
   safety check
   ```
3. Fix issues and re-push

---

## Compliance

### BHIV Security Standards

✅ **Encryption**: AES-256-GCM (FIPS 140-2 compliant)
✅ **Signing**: Ed25519 (NIST approved)
✅ **Watermarking**: Multi-layer (invisible + visible)
✅ **Authentication**: Time-limited runtime keys
✅ **Audit Logging**: Encrypted with KSML binding
✅ **Container Security**: Signed images, SBOM, CVE scanning

### Regulatory Alignment

- **SOC 2 Type II**: Audit logs, access control
- **ISO 27001**: Information security management
- **GDPR**: Data encryption, access logging

---

## Next Steps

### 1. Enable Workflows
```bash
# Push workflows to repository
git add .github/workflows/
git commit -m "feat(security): Add CI/CD security workflows"
git push origin task-10-security-hardening
```

### 2. Configure Secrets
```bash
# In GitHub Settings > Secrets:
# - ARTIFACT_SIGNING_PRIVATE_KEY
# - KSML_TOKEN
# - (Optional) COSIGN_PRIVATE_KEY
```

### 3. Test Workflows
```bash
# Trigger manually
gh workflow run security-gates.yml
gh workflow run security-artifact-signing.yml
gh workflow run security-docker-signing.yml
```

### 4. Enable Branch Protection
```bash
# Settings > Branches > main
# - Require status checks: security-gates
# - Require signed commits
# - Require linear history
```

---

## Support

For security issues or questions:
- **Security Team**: security@bhiv.ai
- **Documentation**: [Task-10-README.md](Task-10-README.md)
- **GitHub Issues**: Use `security` label

---

*Last Updated: 2025-11-06*
*BHIV Security Team*
