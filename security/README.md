# Security Module - Task 10

**BHIV Security Infrastructure for LoRA_TextToVision**

## Overview

This module implements comprehensive security hardening for artifact protection, including:

1. **KSML-bound Encryption** - AES-256-GCM encryption for sensitive data
2. **Cryptographic Signing** - Ed25519 signatures for artifacts
3. **Watermarking** - Deterministic fingerprints tied to BUILD_ID
4. **Runtime Key Validation** - Core-signed time-limited authentication
5. **Provenance Detection** - Tools for detecting unauthorized copies

## Architecture

```
security/
├── __init__.py              # Module exports
├── ksml_encryption.py       # AES-256-GCM encryption
├── artifact_signer.py       # Ed25519 artifact signing
├── watermark.py             # Video watermarking & fingerprinting
└── runtime_validator.py     # Runtime key validation

tools/
└── detect_provenance.py     # Public provenance detection tool
```

## Features

### 1. KSML Encryption

**Purpose:** Encrypt sensitive artifact metadata and audit logs with AES-256-GCM

**Usage:**
```python
from security import ksml_encrypt, ksml_decrypt, ksml_encrypt_json

# Encrypt string
encrypted = ksml_encrypt("Sensitive data")
decrypted = ksml_decrypt(encrypted)

# Encrypt JSON with KSML token binding
data = {"user_id": "123", "prompt": "Ancient temple"}
encrypted_json = ksml_encrypt_json(data, ksml_token="ksml_abc123")
decrypted_json = ksml_decrypt_json(encrypted_json, ksml_token="ksml_abc123")
```

**Key Management:**
- Development: Keys stored in `.ksml_key` (auto-generated, gitignored)
- Production: Use Vault/Task Bank for secure key storage
- Environment variable: `KSML_MASTER_KEY`

### 2. Artifact Signing

**Purpose:** Cryptographic signatures for models, checkpoints, and adapters

**Usage:**
```python
from security import sign_artifact, verify_artifact

# Sign artifact
sig_path = sign_artifact("gurukul_lora.pt", metadata={
    "model_type": "gurukul_lora",
    "version": "1.0.0",
    "build_id": "build_20251106_001"
})

# Verify artifact
is_valid = verify_artifact("gurukul_lora.pt")  # Checks .sig file
```

**CI Integration:**
```bash
# Sign all models in CI
python -c "from security import ArtifactSigner; \
           ArtifactSigner().batch_sign_directory('adapters/gurukul_lora', '*.pt')"
```

**Verification at Runtime:**
```python
from security import verify_artifact

# Verify model before loading
if not verify_artifact("model.pt"):
    raise RuntimeError("Unsigned or tampered model detected!")
```

### 3. Watermarking & Fingerprinting

**Purpose:** Embed detectable watermarks in videos tied to BUILD_ID

**Usage:**
```python
from security import embed_watermark, detect_watermark, compute_fingerprint

# Embed watermark
build_id = os.getenv('BUILD_ID', 'dev_build')
watermarked_video = embed_watermark("output.mp4", build_id=build_id)

# Detect watermark
result = detect_watermark("output.mp4")
if result and result['found']:
    print(f"Build ID: {result['build_id']}")

# Compute content fingerprint
fingerprint = compute_fingerprint("output.mp4", build_id=build_id)
print(f"SHA256: {fingerprint['sha256']}")
```

**Watermark Methods:**
1. **LSB Watermarking** - Least Significant Bit embedding (requires opencv-python)
2. **Metadata Watermarking** - FFmpeg metadata tags (fast, non-invasive)
3. **Perceptual Hashing** - Robust to compression (requires imagehash)

### 4. Runtime Key Validation

**Purpose:** Core-signed runtime keys for worker authentication (12-24h validity)

**Usage:**

**Worker Side (Validation):**
```python
from security import require_runtime_key

# Require valid key or enter restricted mode
try:
    has_key = require_runtime_key(
        runtime_key=os.getenv('RUNTIME_KEY'),
        worker_id="worker_001",
        demo_mode=False  # Strict: require key
    )
except RuntimeError:
    print("No valid runtime key, exiting...")
    sys.exit(1)
```

**Core Side (Issuance):**
```python
from security import RuntimeKeyIssuer

issuer = RuntimeKeyIssuer()
runtime_key = issuer.issue_runtime_key(
    worker_id="worker_001",
    lifetime_hours=24,
    metadata={"region": "us-west"}
)

# Distribute to worker via secure channel
```

**Demo/Restricted Mode:**
```python
# Allow operation without valid key (restricted features)
has_key = require_runtime_key(demo_mode=True)

if not has_key:
    print("RESTRICTED MODE:")
    print("  - Production outputs disabled")
    print("  - Watermarks applied")
    print("  - Quality limited to 720p")
```

### 5. Provenance Detection

**Purpose:** Public tool for detecting file provenance and watermarks

**Usage:**
```bash
# Detect provenance
python tools/detect_provenance.py output.mp4

# JSON output
python tools/detect_provenance.py output.mp4 --json

# Quiet mode (report only)
python tools/detect_provenance.py output.mp4 --quiet
```

**Output:**
```
======================================================================
PROVENANCE REPORT
======================================================================

File: lesson_video_001.mp4
Size: 15,234,567 bytes
Type: .mp4

======================================================================
PROVENANCE STATUS
======================================================================
✅ VERIFIED - File has valid provenance
   Build ID: build_20251106_001

======================================================================
WATERMARK
======================================================================
✅ Watermark detected
   Build ID: build_20251106_001
   Method: metadata_file

======================================================================
CONTENT FINGERPRINT
======================================================================
SHA256:  a1b2c3d4e5f6...
BLAKE2b: x1y2z3a4b5c6...
Perceptual: f9e8d7c6b5a4
```

## Integration

### Video Generation Pipeline

```python
import os
from security import (
    embed_watermark,
    compute_fingerprint,
    ksml_encrypt_json
)
from audit_logger import get_audit_logger

def generate_video_with_security(prompt: str, ksml_token: str):
    # 1. Generate video
    output_path = generate_video(prompt)
    
    # 2. Embed watermark
    build_id = os.getenv('BUILD_ID', 'dev_build')
    watermarked_path = embed_watermark(output_path, build_id=build_id)
    
    # 3. Compute fingerprint
    fingerprint = compute_fingerprint(watermarked_path, build_id=build_id)
    
    # 4. Encrypt sensitive metadata
    encrypted_metadata = ksml_encrypt_json({
        "prompt": prompt,
        "user_id": "user_123",
        "model_version": "1.0.0"
    }, ksml_token=ksml_token)
    
    # 5. Log to audit trail with security fields
    audit_logger = get_audit_logger()
    audit_logger.log_video_generation(
        prompt=prompt,
        output_path=watermarked_path,
        ksml_token={"ksml_token": ksml_token},
        security_metadata={
            "build_id": build_id,
            "artifact_hash": fingerprint['sha256'],
            "watermark_id": build_id,
            "signed": True,
            "encrypted_metadata": encrypted_metadata
        }
    )
    
    return watermarked_path
```

### Model Loading with Verification

```python
from security import verify_artifact
import torch

def load_verified_model(model_path: str):
    # Verify signature before loading
    is_valid = verify_artifact(model_path)
    
    if not is_valid:
        raise RuntimeError(f"Model signature verification failed: {model_path}")
    
    # Load model
    model = torch.load(model_path)
    return model
```

### CI/CD Pipeline Integration

**.github/workflows/security-sign.yml:**
```yaml
name: Security Sign Artifacts

on: [push, pull_request]

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install cryptography
          pip install -r requirements-runtime.txt
      
      - name: Sign artifacts
        env:
          BUILD_ID: ${{ github.sha }}
        run: |
          python -c "
          from security import ArtifactSigner
          signer = ArtifactSigner()
          signer.batch_sign_directory('adapters/gurukul_lora', '*.pt', metadata={'build_id': '$BUILD_ID'})
          "
      
      - name: Upload signatures
        uses: actions/upload-artifact@v3
        with:
          name: signatures
          path: '**/*.sig'
```

## Security Best Practices

### Key Management

1. **Development:**
   - Keys auto-generated in `.signing_keys/` and `.ksml_key`
   - Automatically added to `.gitignore`
   - ⚠️ **NEVER commit keys to version control**

2. **Production:**
   - Use Vault/Task Bank for key storage
   - Set `KSML_MASTER_KEY` environment variable
   - Distribute Core's public key to workers
   - Rotate keys regularly (quarterly minimum)

### Runtime Key Distribution

1. **Secure Channels:**
   - Use HTTPS/TLS for key distribution
   - Core API endpoint requires authentication
   - Time-limited keys (12-24h) minimize exposure

2. **Key Rotation:**
   - Workers request new keys before expiry
   - Graceful fallback to restricted mode if key unavailable

### Artifact Signing

1. **CI/CD:**
   - Sign all artifacts in CI pipeline
   - Store signatures in artifact registry
   - Fail build if signing fails

2. **Runtime:**
   - Verify signatures before loading models
   - Reject unsigned or tampered artifacts
   - Log verification failures to audit trail

### Watermarking

1. **Build ID Management:**
   - Set `BUILD_ID` environment variable per commit
   - Format: `build_YYYYMMDD_NNN` (e.g., `build_20251106_001`)
   - Record in InsightFlow with KSML token

2. **Detection:**
   - Run `detect_provenance.py` on suspicious files
   - Alert Sev-1 if unauthorized copy detected
   - Store fingerprints in BHIV registry

## Environment Variables

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `KSML_MASTER_KEY` | Master encryption key | Production | Auto-generated |
| `BUILD_ID` | Build identifier for watermarks | Yes | `dev_build` |
| `CORE_ENDPOINT` | Core API endpoint for runtime keys | Production | `http://localhost:8080` |
| `RUNTIME_KEY` | Current runtime key | Production | None |

## Testing

### Run Security Tests

```bash
# Test encryption
python security/ksml_encryption.py

# Test signing
python security/artifact_signer.py

# Test watermarking
python security/watermark.py

# Test runtime validation
python security/runtime_validator.py

# Test provenance detection
python tools/detect_provenance.py test_video.mp4
```

### Integration Tests

```bash
# Test full security pipeline
python -c "
from security import *
import tempfile

# Create test artifact
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    f.write(b'test model')
    test_file = f.name

# Sign
sign_artifact(test_file, metadata={'build_id': 'test_001'})

# Verify
assert verify_artifact(test_file), 'Signature verification failed'
print('✅ All security tests passed')
"
```

## Compliance Checklist

Before production deployment, ensure:

- [ ] KSML encryption used for all audit logs
- [ ] Production workers require valid runtime keys
- [ ] All models/adapters signed in CI
- [ ] Signatures verified at model load time
- [ ] Videos have detectable watermarks
- [ ] BUILD_ID recorded in InsightFlow
- [ ] Container images signed with cosign
- [ ] `detect_provenance.py` can extract BUILD_ID
- [ ] Alerting configured for unauthorized copies
- [ ] README includes security handling notes
- [ ] Unit tests verify unsigned model → restricted mode

## Troubleshooting

### Common Issues

**1. "No private key available for signing"**
- Ensure `.signing_keys/private_key.pem` exists
- Or set custom key path: `ArtifactSigner(private_key_path="...")`

**2. "No Core public key found"**
- Create `.signing_keys/core_public_key.pem`
- Or use local key in development (auto-generated)

**3. "Runtime key expired"**
- Request new key: `validator.request_runtime_key(worker_id)`
- Or enable demo mode: `require_runtime_key(demo_mode=True)`

**4. "ffmpeg not found" (watermarking)**
- Install ffmpeg: `sudo apt install ffmpeg`
- Or use fallback metadata file method (automatic)

**5. "Watermark not detected"**
- Check for `.watermark.json` or `.metadata.json` sidecar files
- Ensure ffmpeg/opencv-python installed for extraction

## Dependencies

```bash
pip install cryptography>=41.0.0
pip install imagehash>=4.3.1  # Optional: perceptual hashing
```

## License

Internal BHIV use only. Proprietary security infrastructure.

## Support

For security issues or questions:
- Internal: Contact Security Team
- Production incidents: Create Sev-1 ticket
- Key rotation: Follow Security SOP-001

---

**Created:** November 6, 2025 (Task 10)  
**Version:** 1.0.0  
**Status:** ✅ Complete
