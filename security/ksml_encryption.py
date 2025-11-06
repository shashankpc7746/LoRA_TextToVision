"""
KSML-bound Encryption Module
Provides encryption/decryption for artifact metadata and audit logs
Uses AES-256-GCM with Core-managed keys
"""
import os
import json
import base64
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
import hashlib

class KSMLEncryption:
    """KSML-bound encryption for artifacts and audit logs"""
    
    def __init__(self, key_source: Optional[str] = None):
        """
        Initialize KSML encryption
        
        Args:
            key_source: Path to key file or environment variable name
                       Default: KSML_MASTER_KEY env variable
        """
        self.key = self._load_or_generate_key(key_source)
        self.cipher = AESGCM(self.key)
        
    def _load_or_generate_key(self, key_source: Optional[str] = None) -> bytes:
        """
        Load encryption key from source or generate new one
        
        Priority:
        1. Provided key_source file
        2. KSML_MASTER_KEY environment variable
        3. .ksml_key file in project root
        4. Generate new key (development only)
        """
        # Try provided key source
        if key_source and Path(key_source).exists():
            with open(key_source, 'rb') as f:
                return f.read(32)  # 256-bit key
        
        # Try environment variable
        env_key = os.getenv('KSML_MASTER_KEY')
        if env_key:
            # Derive 256-bit key from env variable
            return self._derive_key(env_key.encode())
        
        # Try .ksml_key file
        key_file = Path('.ksml_key')
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read(32)
        
        # Generate new key (development only - should use Vault in production)
        print("⚠️  WARNING: Generating new KSML key. Use Vault/Task Bank in production!")
        new_key = AESGCM.generate_key(bit_length=256)
        
        # Save to .ksml_key for development
        with open(key_file, 'wb') as f:
            f.write(new_key)
        
        # Add to .gitignore
        gitignore = Path('.gitignore')
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                content = f.read()
            if '.ksml_key' not in content:
                with open(gitignore, 'a') as f:
                    f.write('\n# KSML encryption key (DO NOT COMMIT)\n.ksml_key\n')
        
        return new_key
    
    def _derive_key(self, password: bytes, salt: Optional[bytes] = None) -> bytes:
        """Derive 256-bit key from password using PBKDF2"""
        if salt is None:
            salt = b'ksml_bhiv_salt_v1'  # Fixed salt for deterministic derivation
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)
    
    def encrypt(self, plaintext: str, associated_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Encrypt plaintext with KSML key
        
        Args:
            plaintext: String to encrypt
            associated_data: Optional metadata to authenticate (not encrypted)
        
        Returns:
            Base64-encoded encrypted data with nonce
        """
        # Convert plaintext to bytes
        plaintext_bytes = plaintext.encode('utf-8')
        
        # Generate random nonce
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        # Prepare associated data
        aad = None
        if associated_data:
            aad = json.dumps(associated_data, sort_keys=True).encode('utf-8')
        
        # Encrypt
        ciphertext = self.cipher.encrypt(nonce, plaintext_bytes, aad)
        
        # Package: nonce + ciphertext
        encrypted_package = nonce + ciphertext
        
        # Return base64-encoded
        return base64.b64encode(encrypted_package).decode('utf-8')
    
    def decrypt(self, encrypted_data: str, associated_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Decrypt KSML-encrypted data
        
        Args:
            encrypted_data: Base64-encoded encrypted package
            associated_data: Optional metadata to verify (must match encryption)
        
        Returns:
            Decrypted plaintext string
        """
        # Decode base64
        encrypted_package = base64.b64decode(encrypted_data)
        
        # Extract nonce and ciphertext
        nonce = encrypted_package[:12]
        ciphertext = encrypted_package[12:]
        
        # Prepare associated data
        aad = None
        if associated_data:
            aad = json.dumps(associated_data, sort_keys=True).encode('utf-8')
        
        # Decrypt
        plaintext_bytes = self.cipher.decrypt(nonce, ciphertext, aad)
        
        # Return plaintext
        return plaintext_bytes.decode('utf-8')
    
    def encrypt_file(self, input_path: str, output_path: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Encrypt entire file with KSML key
        
        Args:
            input_path: Path to file to encrypt
            output_path: Path for encrypted file (default: input_path + '.encrypted')
            metadata: Optional metadata to include
        
        Returns:
            Path to encrypted file
        """
        if output_path is None:
            output_path = input_path + '.encrypted'
        
        # Read file
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        # Generate nonce
        nonce = os.urandom(12)
        
        # Prepare metadata
        file_metadata = {
            'original_filename': Path(input_path).name,
            'original_size': len(plaintext),
            'encrypted_at': str(hashlib.sha256(plaintext).hexdigest()[:16])
        }
        if metadata:
            file_metadata.update(metadata)
        
        aad = json.dumps(file_metadata, sort_keys=True).encode('utf-8')
        
        # Encrypt
        ciphertext = self.cipher.encrypt(nonce, plaintext, aad)
        
        # Write encrypted file: metadata_length(4 bytes) + metadata + nonce + ciphertext
        with open(output_path, 'wb') as f:
            metadata_bytes = aad
            f.write(len(metadata_bytes).to_bytes(4, 'big'))
            f.write(metadata_bytes)
            f.write(nonce)
            f.write(ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """
        Decrypt KSML-encrypted file
        
        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted file (default: remove .encrypted extension)
        
        Returns:
            Tuple of (output_path, metadata)
        """
        # Read encrypted file
        with open(input_path, 'rb') as f:
            # Read metadata length
            metadata_length = int.from_bytes(f.read(4), 'big')
            
            # Read metadata
            metadata_bytes = f.read(metadata_length)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Read nonce
            nonce = f.read(12)
            
            # Read ciphertext
            ciphertext = f.read()
        
        # Decrypt
        plaintext = self.cipher.decrypt(nonce, ciphertext, metadata_bytes)
        
        # Determine output path
        if output_path is None:
            if input_path.endswith('.encrypted'):
                output_path = input_path[:-10]
            else:
                output_path = input_path + '.decrypted'
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path, metadata
    
    def encrypt_json(self, data: Dict[str, Any], ksml_token: Optional[str] = None) -> str:
        """
        Encrypt JSON data with optional KSML token binding
        
        Args:
            data: Dictionary to encrypt
            ksml_token: Optional KSML token to bind
        
        Returns:
            Base64-encoded encrypted JSON
        """
        plaintext = json.dumps(data, sort_keys=True)
        
        associated_data = {}
        if ksml_token:
            associated_data['ksml_token'] = ksml_token
        
        return self.encrypt(plaintext, associated_data)
    
    def decrypt_json(self, encrypted_data: str, ksml_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Decrypt KSML-encrypted JSON
        
        Args:
            encrypted_data: Base64-encoded encrypted JSON
            ksml_token: Optional KSML token to verify
        
        Returns:
            Decrypted dictionary
        """
        associated_data = {}
        if ksml_token:
            associated_data['ksml_token'] = ksml_token
        
        plaintext = self.decrypt(encrypted_data, associated_data)
        return json.loads(plaintext)


# Convenience functions
_ksml_cipher = None

def get_ksml_cipher() -> KSMLEncryption:
    """Get global KSML cipher instance"""
    global _ksml_cipher
    if _ksml_cipher is None:
        _ksml_cipher = KSMLEncryption()
    return _ksml_cipher


def ksml_encrypt(plaintext: str, associated_data: Optional[Dict[str, Any]] = None) -> str:
    """Encrypt plaintext with KSML key"""
    return get_ksml_cipher().encrypt(plaintext, associated_data)


def ksml_decrypt(encrypted_data: str, associated_data: Optional[Dict[str, Any]] = None) -> str:
    """Decrypt KSML-encrypted data"""
    return get_ksml_cipher().decrypt(encrypted_data, associated_data)


def ksml_encrypt_json(data: Dict[str, Any], ksml_token: Optional[str] = None) -> str:
    """Encrypt JSON data with KSML binding"""
    return get_ksml_cipher().encrypt_json(data, ksml_token)


def ksml_decrypt_json(encrypted_data: str, ksml_token: Optional[str] = None) -> Dict[str, Any]:
    """Decrypt KSML-encrypted JSON"""
    return get_ksml_cipher().decrypt_json(encrypted_data, ksml_token)


if __name__ == "__main__":
    # Test encryption
    cipher = KSMLEncryption()
    
    # Test string encryption
    plaintext = "Sensitive KSML data"
    encrypted = cipher.encrypt(plaintext)
    decrypted = cipher.decrypt(encrypted)
    
    print(f"Original: {plaintext}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"✅ String encryption test: {'PASSED' if plaintext == decrypted else 'FAILED'}")
    
    # Test JSON encryption with KSML token
    data = {
        "user_id": "user_123",
        "prompt": "Ancient Gurukul classroom",
        "model": "gurukul_lora.pt",
        "timestamp": "2025-11-06T10:00:00Z"
    }
    ksml_token = "ksml_token_abc123"
    
    encrypted_json = cipher.encrypt_json(data, ksml_token)
    decrypted_json = cipher.decrypt_json(encrypted_json, ksml_token)
    
    print(f"\nOriginal JSON: {data}")
    print(f"Encrypted: {encrypted_json[:50]}...")
    print(f"Decrypted JSON: {decrypted_json}")
    print(f"✅ JSON encryption test: {'PASSED' if data == decrypted_json else 'FAILED'}")
