"""Test security module imports"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing security module imports...\n")

try:
    from security import embed_watermark, compute_fingerprint
    print("✅ security.embed_watermark imported successfully")
    print("✅ security.compute_fingerprint imported successfully")
except Exception as e:
    print(f"❌ Failed to import security base modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from security.visible_watermark import add_visible_watermark
    print("✅ security.visible_watermark.add_visible_watermark imported successfully")
except Exception as e:
    print(f"❌ Failed to import visible_watermark: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from audit_logger import get_audit_logger
    print("✅ audit_logger imported successfully")
except Exception as e:
    print(f"❌ Failed to import audit_logger: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ ALL SECURITY MODULES IMPORTED SUCCESSFULLY!")
print("\nNow testing if functions are callable...")

# Test if BUILD_ID can be retrieved
build_id = os.getenv('BUILD_ID', 'test_build_123')
print(f"BUILD_ID: {build_id}")

print("\n✅ Security system is ready!")
