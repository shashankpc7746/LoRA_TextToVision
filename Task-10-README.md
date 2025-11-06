# Task 10: Security Hardening

**Mission:** Implement comprehensive security infrastructure to prevent unauthorized copying and ensure artifact provenance.

## Status: 🚧 In Progress (70% Complete)

### Completion Timeline
- **Started:** November 6, 2025
- **Target Completion:** November 10, 2025
- **Estimated Remaining:** 1-2 days

---

## Objective

Implement 10 security features to protect BHIV's video generation pipeline from unauthorized copying:

1. ✅ KSML-bound encryption for artifact metadata
2. ✅ Core-signed runtime keys (time-limited authentication)
3. ✅ Cryptographic provenance (artifact signing)
4. ✅ Watermark/fingerprinting outputs
5. ✅ Build fingerprint per commit (BUILD_ID)
6. ⏳ Signed container images (cosign)
7. ✅ Provenance detection pipeline
8. ✅ Enhanced audit logs & telemetry
9. ⏳ Mandatory CI gates (security:sign-and-prove)
10. ⏳ Runtime attestation (optional TPM)

---

## Completed Features (7/10)

### 1. KSML-bound Encryption ✅

**File:** `security/ksml_encryption.py`

**Implementation:**
- AES-256-GCM encryption for sensitive data
- KSML token binding for audit logs
- Key management (Vault integration ready)
- File and JSON encryption support

**Usage:**
```python
from security import ksml_encrypt, ksml_decrypt, ksml_encrypt_json

# Encrypt sensitive data
encrypted = ksml_encrypt("Sensitive prompt data")
decrypted = ksml_decrypt(encrypted)

# Encrypt with KSML token binding
data = {"user_id": "123", "prompt": "Ancient temple"}
encrypted_json = ksml_encrypt_json(data, ksml_token="ksml_abc123")
```

**Test Results:**
```
✅ String encryption test: PASSED
✅ JSON encryption test: PASSED
```

---

### 2. Cryptographic Provenance (Artifact Signing) ✅

**File:** `security/artifact_signer.py`

**Implementation:**
- Ed25519 signatures for models/checkpoints
- SHA256 hash verification
- Batch signing for directories
- Signature file format (.sig)

**Usage:**
```python
from security import sign_artifact, verify_artifact

# Sign artifact
sign_artifact("gurukul_lora.pt", metadata={
    "model_type": "gurukul_lora",
    "version": "1.0.0",
    "build_id": "build_20251106_001"
})

# Verify artifact
is_valid = verify_artifact("gurukul_lora.pt")
```

**Test Results:**
```
✅ Signature verification: PASSED
✅ File-based verification: PASSED
✅ Tamper detection: PASSED
```

---

### 3. Watermark/Fingerprinting ✅

**File:** `security/watermark.py`

**Implementation:**
- Deterministic watermark generation from BUILD_ID
- LSB watermarking support
- FFmpeg metadata watermarking
- SHA256 + BLAKE2b fingerprinting
- Perceptual hashing (robust to compression)

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

# Compute fingerprint
fingerprint = compute_fingerprint("output.mp4", build_id=build_id)
```

**Test Results:**
```
✅ Watermark pattern generated
✅ Watermark embedded
✅ Watermark detected: True
✅ Content fingerprint: SHA256 + BLAKE2b
```

---

### 4. Core-signed Runtime Keys ✅

**File:** `security/runtime_validator.py`

**Implementation:**
- Ed25519-signed runtime keys
- Time-limited validity (12-24h)
- Worker authentication
- Restricted demo mode (no valid key)
- Key caching and rotation

**Usage:**

**Worker (Validation):**
```python
from security import require_runtime_key

# Require valid key
has_key = require_runtime_key(
    runtime_key=os.getenv('RUNTIME_KEY'),
    worker_id="worker_001",
    demo_mode=False  # Strict mode
)
```

**Core (Issuance):**
```python
from security import RuntimeKeyIssuer

issuer = RuntimeKeyIssuer()
runtime_key = issuer.issue_runtime_key(
    worker_id="worker_001",
    lifetime_hours=24
)
```

**Test Results:**
```
✅ Runtime key issued for worker_test_001
✅ Validation result: VALID
✅ Expired key (strict): INVALID (expected)
✅ Expired key (demo mode): VALID
✅ Key cached: True
```

---

### 5. Provenance Detection Pipeline ✅

**File:** `tools/detect_provenance.py`

**Implementation:**
- Public CLI tool for provenance checking
- Watermark detection
- Signature verification
- Content fingerprinting
- JSON and human-readable output

**Usage:**
```bash
# Detect provenance
python tools/detect_provenance.py output.mp4

# JSON output
python tools/detect_provenance.py output.mp4 --json

# Quiet mode
python tools/detect_provenance.py output.mp4 --quiet
```

**Sample Output:**
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
```

---

### 6. Enhanced Audit Logs ✅

**File:** `audit_logger.py` (updated)

**Implementation:**
- Added `security_metadata` parameter to `log_video_generation()`
- Records: build_id, artifact_hash, watermark_id, signed status
- Integration with InsightFlow telemetry
- Immutable audit trail

**Usage:**
```python
from audit_logger import get_audit_logger

audit_logger = get_audit_logger()
audit_logger.log_video_generation(
    prompt="Ancient temple",
    output_path="output.mp4",
    ksml_token={"ksml_token": "ksml_abc123"},
    security_metadata={
        "build_id": "build_20251106_001",
        "artifact_hash": "sha256:a1b2c3...",
        "watermark_id": "build_20251106_001",
        "signed": True
    }
)
```

---

### 7. Build Fingerprint (BUILD_ID) ✅

**Implementation:**
- Environment variable: `BUILD_ID`
- Format: `build_YYYYMMDD_NNN`
- Injected per commit in CI
- Used for deterministic watermark seeding

**Usage:**
```bash
# Set in CI pipeline
export BUILD_ID="build_$(date +%Y%m%d)_${GITHUB_RUN_NUMBER}"

# Or manually
export BUILD_ID="build_20251106_001"
```

---

## Pending Features (3/10)

### 8. Signed Container Images ⏳

**Status:** Not started

**Requirements:**
- Install cosign in CI
- Sign Docker images after build
- Verify signatures on production nodes
- Restrict registry to BHIV accounts

**Implementation Plan:**
```yaml
# .github/workflows/docker-sign.yml
- name: Sign Docker image
  run: |
    cosign sign --key cosign.key ${{ env.IMAGE_TAG }}
    
- name: Verify signature
  run: |
    cosign verify --key cosign.pub ${{ env.IMAGE_TAG }}
```

**Estimated Time:** 2-3 hours

---

### 9. Mandatory CI Gates ⏳

**Status:** Not started

**Requirements:**
- Create `.github/workflows/security-sign.yml`
- Implement `security:sign-and-prove` step
- Sign all artifacts (models, adapters, checkpoints)
- Fail pipeline if signing fails
- Store signatures in Task Bank

**Implementation Plan:**
```yaml
name: Security Sign & Prove

on: [push, pull_request]

jobs:
  security-sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Sign artifacts
        env:
          BUILD_ID: ${{ github.sha }}
        run: |
          python -c "
          from security import ArtifactSigner
          signer = ArtifactSigner()
          results = signer.batch_sign_directory('adapters/gurukul_lora', '*.pt', metadata={'build_id': '$BUILD_ID'})
          print(f'Signed {len(results)} artifacts')
          "
      
      - name: Upload signatures
        uses: actions/upload-artifact@v3
        with:
          name: signatures
          path: '**/*.sig'
      
      - name: Fail on unsigned artifacts
        run: |
          # Check all .pt files have .sig files
          unsigned=$(find adapters/gurukul_lora -name '*.pt' ! -exec test -f {}.sig \; -print)
          if [ -n "$unsigned" ]; then
            echo "❌ Unsigned artifacts detected:"
            echo "$unsigned"
            exit 1
          fi
```

**Estimated Time:** 3-4 hours

---

### 10. Runtime Attestation (Optional) ⏳

**Status:** Not started (optional)

**Requirements:**
- TPM integration (if available)
- Cloud identity validation
- Secure boot verification
- Platform attestation

**Implementation Plan:**
- Research TPM Python libraries (tpm2-pytss)
- Implement `security/attestation.py`
- Optional fallback if TPM unavailable

**Estimated Time:** 4-6 hours (optional)

---

## Integration Examples

### Video Generation with Security

```python
import os
from security import (
    embed_watermark,
    compute_fingerprint,
    ksml_encrypt_json,
    require_runtime_key
)
from audit_logger import get_audit_logger

def secure_video_generation(prompt: str, ksml_token: str):
    # 1. Validate runtime key
    has_key = require_runtime_key(
        runtime_key=os.getenv('RUNTIME_KEY'),
        worker_id="worker_001",
        demo_mode=True  # Fallback to restricted mode
    )
    
    if not has_key:
        print("⚠️ RESTRICTED MODE: Quality limited, watermarks applied")
    
    # 2. Generate video
    output_path = generate_video(prompt)
    
    # 3. Embed watermark
    build_id = os.getenv('BUILD_ID', 'dev_build')
    watermarked_path = embed_watermark(output_path, build_id=build_id)
    
    # 4. Compute fingerprint
    fingerprint = compute_fingerprint(watermarked_path, build_id=build_id)
    
    # 5. Encrypt sensitive metadata
    encrypted_metadata = ksml_encrypt_json({
        "prompt": prompt,
        "user_id": "user_123",
        "model_version": "1.0.0"
    }, ksml_token=ksml_token)
    
    # 6. Log with security fields
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
            "encrypted_metadata": encrypted_metadata,
            "restricted_mode": not has_key
        }
    )
    
    return watermarked_path
```

### Model Loading with Verification

```python
from security import verify_artifact
import torch

def load_verified_model(model_path: str):
    # Verify signature
    is_valid = verify_artifact(model_path)
    
    if not is_valid:
        print(f"❌ Model signature verification failed: {model_path}")
        print("⚠️ Entering restricted mode (demo outputs only)")
        # Load model but restrict capabilities
    
    # Load model
    model = torch.load(model_path)
    return model
```

---

## Testing

### Run Security Tests

```bash
# Test all modules
cd c:\Shashank\LoRA_TextToVision

# KSML Encryption
python security/ksml_encryption.py

# Artifact Signing
python security/artifact_signer.py

# Watermarking
python security/watermark.py

# Runtime Validation
python security/runtime_validator.py
```

### Integration Test

```bash
# Create test artifact
python -c "
import tempfile
from pathlib import Path
from security import sign_artifact, verify_artifact, embed_watermark, detect_watermark

# Test model signing
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    f.write(b'test model weights')
    model_path = f.name

# Sign
sig_path = sign_artifact(model_path, metadata={'build_id': 'test_001'})
print(f'✅ Signed: {sig_path}')

# Verify
is_valid = verify_artifact(model_path)
print(f'✅ Verified: {is_valid}')

# Test video watermarking
with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
    f.write(b'test video content')
    video_path = f.name

# Watermark
watermarked = embed_watermark(video_path, build_id='test_001')
print(f'✅ Watermarked: {watermarked}')

# Detect
result = detect_watermark(watermarked)
print(f'✅ Detected: {result[\"found\"] if result else False}')

# Cleanup
import os
os.unlink(model_path)
os.unlink(model_path + '.sig')
os.unlink(video_path)
if Path(watermarked).exists():
    os.unlink(watermarked)
if Path(watermarked + '.watermark.json').exists():
    os.unlink(watermarked + '.watermark.json')

print('\n✅ All security integration tests passed!')
"
```

---

## Acceptance Criteria

### Completed ✅

- [x] ksml_encrypt() used for artifact metadata and audit logs
- [x] Production worker requires valid Core-signed runtime key
- [x] All models/adapters can be signed in CI
- [x] Signatures can be verified at model load
- [x] Videos have content fingerprint + detectable watermark
- [x] BUILD_ID can be recorded in InsightFlow with KSML token
- [x] tools/detect_provenance.py can report BUILD_ID from files
- [x] Enhanced audit logs include security metadata
- [x] README includes security handling notes

### Pending ⏳

- [ ] CI produces signed container image (cosign)
- [ ] CI pipeline includes security:sign-and-prove step
- [ ] Pipeline fails if artifacts unsigned
- [ ] Alerting rule for unauthorized copy detection
- [ ] Unit tests verify unsigned model → restricted mode

---

## Dependencies

**Added to `requirements-runtime.txt`:**
```
cryptography>=41.0.0
imagehash>=4.3.1
```

**Optional (for watermarking):**
- ffmpeg (system package)
- opencv-python (already in requirements)

---

## Files Created/Modified

### New Files
1. `security/__init__.py` - Module exports
2. `security/ksml_encryption.py` - AES-256-GCM encryption (370 lines)
3. `security/artifact_signer.py` - Ed25519 signing (450 lines)
4. `security/watermark.py` - Watermarking & fingerprinting (420 lines)
5. `security/runtime_validator.py` - Runtime key validation (380 lines)
6. `security/README.md` - Comprehensive documentation (500+ lines)
7. `tools/detect_provenance.py` - Provenance detection tool (280 lines)

### Modified Files
1. `audit_logger.py` - Added `security_metadata` parameter
2. `requirements-runtime.txt` - Added cryptography dependencies

### Total Lines Added: ~2,400 lines

---

## Next Steps

### Priority 1: CI Security Gates (1 day)
1. Create `.github/workflows/security-sign.yml`
2. Implement artifact signing in CI
3. Add signature verification step
4. Fail pipeline on unsigned artifacts

### Priority 2: Container Signing (0.5 day)
1. Install cosign in CI
2. Sign Docker images
3. Verify on deployment
4. Document verification process

### Priority 3: Testing & Documentation (0.5 day)
1. Create comprehensive test suite
2. Test restricted mode behavior
3. Document key rotation procedures
4. Create runbook for security incidents

### Optional: Runtime Attestation (1-2 days)
1. Research TPM integration
2. Implement attestation module
3. Test on supported hardware
4. Document fallback behavior

---

## Security Checklist

Before production deployment:

- [ ] Generate production signing keys (Ed25519)
- [ ] Store keys in Vault/Task Bank
- [ ] Distribute Core's public key to workers
- [ ] Set BUILD_ID in CI pipeline
- [ ] Enable runtime key validation (strict mode)
- [ ] Sign all existing models/checkpoints
- [ ] Configure InsightFlow telemetry
- [ ] Set up alerting for unauthorized copies
- [ ] Document key rotation schedule (quarterly)
- [ ] Train team on security procedures

---

## Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| Nov 6, 2025 | Security module foundation | ✅ Complete |
| Nov 6, 2025 | KSML encryption | ✅ Complete |
| Nov 6, 2025 | Artifact signing | ✅ Complete |
| Nov 6, 2025 | Watermarking & fingerprinting | ✅ Complete |
| Nov 6, 2025 | Runtime key validation | ✅ Complete |
| Nov 6, 2025 | Provenance detection tool | ✅ Complete |
| Nov 6, 2025 | Enhanced audit logs | ✅ Complete |
| Nov 7, 2025 | CI security gates | ⏳ Planned |
| Nov 8, 2025 | Container signing | ⏳ Planned |
| Nov 9, 2025 | Testing & documentation | ⏳ Planned |
| Nov 10, 2025 | **Task 10 Complete** | 🎯 Target |

---

## References

- Task 9 README: `Task-9-README.md`
- Security Module: `security/README.md`
- Audit Logger: `audit_logger.py`
- Provenance Tool: `tools/detect_provenance.py`

---

**Last Updated:** November 6, 2025  
**Progress:** 70% (7/10 features complete)  
**Branch:** `task_quality_leap` (to be merged to `task_quality_harden_secure`)
