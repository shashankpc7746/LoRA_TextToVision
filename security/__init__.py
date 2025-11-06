"""
BHIV Security Module
Encryption, signing, watermarking, and runtime validation for artifacts
"""

from .ksml_encryption import (
    KSMLEncryption,
    ksml_encrypt,
    ksml_decrypt,
    ksml_encrypt_json,
    ksml_decrypt_json,
    get_ksml_cipher
)

from .artifact_signer import (
    ArtifactSigner,
    sign_artifact,
    verify_artifact,
    get_artifact_signer
)

from .watermark import (
    VideoWatermarker,
    ContentFingerprinter,
    embed_watermark,
    detect_watermark,
    compute_fingerprint
)

from .runtime_validator import (
    RuntimeKeyValidator,
    RuntimeKeyIssuer,
    validate_runtime_key,
    require_runtime_key,
    get_runtime_validator
)

from .visible_watermark import (
    VisibleWatermarker,
    add_visible_watermark
)

__all__ = [
    # KSML Encryption
    'KSMLEncryption',
    'ksml_encrypt',
    'ksml_decrypt',
    'ksml_encrypt_json',
    'ksml_decrypt_json',
    'get_ksml_cipher',
    
    # Artifact Signing
    'ArtifactSigner',
    'sign_artifact',
    'verify_artifact',
    'get_artifact_signer',
    
    # Watermarking
    'VideoWatermarker',
    'ContentFingerprinter',
    'embed_watermark',
    'detect_watermark',
    'compute_fingerprint',
    
    # Runtime Validation
    'RuntimeKeyValidator',
    'RuntimeKeyIssuer',
    'validate_runtime_key',
    'require_runtime_key',
    'get_runtime_validator',
    
    # Visible Watermarking
    'VisibleWatermarker',
    'add_visible_watermark',
]

__version__ = '1.0.0'
