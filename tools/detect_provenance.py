"""
Provenance Detection Tool
Public script for checking file provenance and watermarks
Usage: python detect_provenance.py <file_path>
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.watermark import detect_watermark, ContentFingerprinter
from security.artifact_signer import get_artifact_signer


def detect_provenance(file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Detect provenance of a file (video, model, checkpoint)
    
    Args:
        file_path: Path to file to check
        verbose: Print detailed output
    
    Returns:
        Provenance information dictionary
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        return {
            'found': False,
            'error': 'File not found'
        }
    
    result = {
        'file': str(file_path_obj.name),
        'file_path': str(file_path_obj.absolute()),
        'file_size': file_path_obj.stat().st_size,
        'file_type': file_path_obj.suffix.lower(),
    }
    
    # Check for watermark (videos)
    if file_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        if verbose:
            print(f"\n🔍 Checking video watermark...")
        
        watermark_result = detect_watermark(file_path)
        
        if watermark_result and watermark_result.get('found'):
            result['watermark'] = {
                'found': True,
                'build_id': watermark_result.get('build_id'),
                'detection_method': watermark_result.get('detection_method'),
            }
            
            if verbose:
                print(f"✅ Watermark detected!")
                print(f"   Build ID: {watermark_result.get('build_id')}")
                print(f"   Method: {watermark_result.get('detection_method')}")
        else:
            result['watermark'] = {'found': False}
            if verbose:
                print(f"❌ No watermark detected")
    
    # Compute content fingerprint
    if verbose:
        print(f"\n🔍 Computing content fingerprint...")
    
    try:
        fingerprint = ContentFingerprinter.create_fingerprint_record(
            file_path, 
            result.get('watermark', {}).get('build_id', 'unknown')
        )
        
        result['fingerprint'] = {
            'sha256': fingerprint['sha256'],
            'blake2b': fingerprint['blake2b'],
        }
        
        if 'perceptual_hash' in fingerprint:
            result['fingerprint']['perceptual_hash'] = fingerprint['perceptual_hash']
        
        if verbose:
            print(f"✅ Content fingerprint computed")
            print(f"   SHA256: {fingerprint['sha256'][:32]}...")
            print(f"   BLAKE2b: {fingerprint['blake2b'][:32]}...")
            if 'perceptual_hash' in fingerprint:
                print(f"   Perceptual: {fingerprint['perceptual_hash']}")
    except Exception as e:
        result['fingerprint'] = {'error': str(e)}
        if verbose:
            print(f"❌ Fingerprint error: {e}")
    
    # Check for signature (models, checkpoints)
    if file_path_obj.suffix.lower() in ['.pt', '.pth', '.safetensors', '.ckpt']:
        if verbose:
            print(f"\n🔍 Checking artifact signature...")
        
        sig_path = str(file_path) + '.sig'
        
        if Path(sig_path).exists():
            try:
                signer = get_artifact_signer()
                is_valid, sig_data = signer.verify_from_file(file_path)
                
                result['signature'] = {
                    'found': True,
                    'valid': is_valid,
                    'signed_at': sig_data.get('signed_at') if sig_data else None,
                    'metadata': sig_data.get('metadata') if sig_data else None,
                }
                
                if verbose:
                    print(f"{'✅' if is_valid else '❌'} Signature {'VALID' if is_valid else 'INVALID'}")
                    if sig_data:
                        print(f"   Signed at: {sig_data.get('signed_at')}")
                        if sig_data.get('metadata'):
                            print(f"   Metadata: {json.dumps(sig_data['metadata'], indent=6)}")
            except Exception as e:
                result['signature'] = {
                    'found': True,
                    'valid': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"❌ Signature verification error: {e}")
        else:
            result['signature'] = {'found': False}
            if verbose:
                print(f"❌ No signature file found (.sig)")
    
    # Determine provenance status
    has_watermark = result.get('watermark', {}).get('found', False)
    has_valid_signature = result.get('signature', {}).get('valid', False)
    
    if has_watermark or has_valid_signature:
        result['provenance'] = 'verified'
        result['build_id'] = (
            result.get('watermark', {}).get('build_id') or
            result.get('signature', {}).get('metadata', {}).get('build_id')
        )
    else:
        result['provenance'] = 'unknown'
    
    return result


def format_provenance_report(provenance: Dict[str, Any]) -> str:
    """Format provenance data as readable report"""
    lines = []
    lines.append("=" * 70)
    lines.append("PROVENANCE REPORT")
    lines.append("=" * 70)
    
    lines.append(f"\nFile: {provenance['file']}")
    lines.append(f"Size: {provenance['file_size']:,} bytes")
    lines.append(f"Type: {provenance['file_type']}")
    
    lines.append(f"\n{'=' * 70}")
    lines.append("PROVENANCE STATUS")
    lines.append(f"{'=' * 70}")
    
    status = provenance.get('provenance', 'unknown')
    if status == 'verified':
        lines.append("✅ VERIFIED - File has valid provenance")
        if 'build_id' in provenance:
            lines.append(f"   Build ID: {provenance['build_id']}")
    else:
        lines.append("❌ UNKNOWN - No provenance information found")
    
    # Watermark section
    if 'watermark' in provenance:
        lines.append(f"\n{'=' * 70}")
        lines.append("WATERMARK")
        lines.append(f"{'=' * 70}")
        
        if provenance['watermark'].get('found'):
            lines.append("✅ Watermark detected")
            lines.append(f"   Build ID: {provenance['watermark'].get('build_id')}")
            lines.append(f"   Method: {provenance['watermark'].get('detection_method')}")
        else:
            lines.append("❌ No watermark detected")
    
    # Signature section
    if 'signature' in provenance:
        lines.append(f"\n{'=' * 70}")
        lines.append("CRYPTOGRAPHIC SIGNATURE")
        lines.append(f"{'=' * 70}")
        
        if provenance['signature'].get('found'):
            is_valid = provenance['signature'].get('valid')
            lines.append(f"{'✅' if is_valid else '❌'} Signature {'VALID' if is_valid else 'INVALID'}")
            
            if provenance['signature'].get('signed_at'):
                lines.append(f"   Signed at: {provenance['signature']['signed_at']}")
            
            if provenance['signature'].get('metadata'):
                lines.append(f"   Metadata: {json.dumps(provenance['signature']['metadata'], indent=6)}")
        else:
            lines.append("❌ No signature found")
    
    # Fingerprint section
    if 'fingerprint' in provenance:
        lines.append(f"\n{'=' * 70}")
        lines.append("CONTENT FINGERPRINT")
        lines.append(f"{'=' * 70}")
        
        if 'error' in provenance['fingerprint']:
            lines.append(f"❌ Error: {provenance['fingerprint']['error']}")
        else:
            lines.append(f"SHA256:  {provenance['fingerprint']['sha256']}")
            lines.append(f"BLAKE2b: {provenance['fingerprint']['blake2b']}")
            
            if 'perceptual_hash' in provenance['fingerprint']:
                lines.append(f"Perceptual: {provenance['fingerprint']['perceptual_hash']}")
    
    lines.append(f"\n{'=' * 70}")
    
    return '\n'.join(lines)


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python detect_provenance.py <file_path> [--json]")
        print("\nDetect provenance of videos, models, and checkpoints")
        print("\nOptions:")
        print("  --json    Output result as JSON")
        print("  --quiet   Suppress detailed output")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_json = '--json' in sys.argv
    verbose = '--quiet' not in sys.argv
    
    # Detect provenance
    provenance = detect_provenance(file_path, verbose=verbose and not output_json)
    
    # Output result
    if output_json:
        print(json.dumps(provenance, indent=2))
    else:
        if verbose:
            print()  # Blank line after detection output
        print(format_provenance_report(provenance))


if __name__ == "__main__":
    main()
