#!/usr/bin/env python3
"""
Test Yotta Fallback Validation for Task-6
Tests forcing routing to Yotta tier and validates signed URL generation
"""

import requests
import json
import time
from pathlib import Path

def test_yotta_fallback():
    """Test forcing routing to Yotta tier"""

    print("🧪 Testing Yotta Fallback Validation (Task-6)")
    print("=" * 50)

    # Test data
    test_request = {
        "prompt": "Test Yotta fallback validation",
        "style": "realistic",
        "target_quality": "balanced",
        "max_cost_usd": 1.0,  # Allow higher cost for Yotta
        "max_latency_sec": 600,  # Allow longer latency
        "force_tier": "yotta"  # Force Yotta routing
    }

    try:
        # Make request to force Yotta tier
        print("📤 Making request with force_tier=yotta...")
        response = requests.post(
            "http://localhost:8001/ttv/generate",
            json=test_request,
            timeout=30
        )

        print(f"📥 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful")

            # Check if tier was forced to yotta
            selected_tier = result.get("selected_tier")
            print(f"🎯 Selected tier: {selected_tier}")

            if selected_tier == "yotta":
                print("✅ SUCCESS: Request was correctly routed to Yotta tier")

                # Check for video_url
                video_url = result.get("video_url")
                if video_url:
                    print(f"🎬 Video URL: {video_url}")

                    # Try to access the video URL (should be signed URL)
                    try:
                        video_response = requests.get(video_url, timeout=10)
                        if video_response.status_code == 200:
                            print("✅ SUCCESS: Video URL is accessible (signed URL working)")
                        else:
                            print(f"⚠️ WARNING: Video URL returned status {video_response.status_code}")
                    except Exception as e:
                        print(f"❌ ERROR: Could not access video URL: {e}")
                else:
                    print("⚠️ WARNING: No video_url in response")

                # Check routing decision details
                routing_decision = result.get("routing_decision", {})
                print(f"💰 Estimated cost: ${routing_decision.get('estimated_cost', 'N/A')}")
                print(f"⏱️ Estimated latency: {routing_decision.get('estimated_latency', 'N/A')}ms")

                return True

            else:
                print(f"❌ FAILED: Expected 'yotta' tier, got '{selected_tier}'")
                return False

        else:
            print(f"❌ FAILED: Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: Request exception: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

def test_nas_signed_url():
    """Test NAS signed URL generation"""
    print("\n🔗 Testing NAS Signed URL Generation...")

    try:
        # First, try to write a test file to NAS
        test_filename = f"test_yotta_validation_{int(time.time())}.txt"
        test_content = "Test file for Yotta fallback validation"

        # Write to NAS
        write_response = requests.post(
            "http://localhost:8001/ttv/nas/write",
            json={
                "filename": test_filename,
                "local_path": None,  # We'll create content on the fly
                "content": test_content
            },
            timeout=10
        )

        if write_response.status_code == 200:
            print("✅ File written to NAS successfully")

            # Get signed URL
            signed_url_response = requests.get(
                f"http://localhost:8001/ttv/nas/signed-url/{test_filename}",
                timeout=10
            )

            if signed_url_response.status_code == 200:
                signed_url_data = signed_url_response.json()
                signed_url = signed_url_data.get("signed_url")

                if signed_url:
                    print("✅ Signed URL generated successfully")
                    print(f"🔗 Signed URL: {signed_url[:100]}...")

                    # Try to access via signed URL
                    access_response = requests.get(signed_url, timeout=10)
                    if access_response.status_code == 200:
                        retrieved_content = access_response.text
                        if retrieved_content == test_content:
                            print("✅ Signed URL access successful - content matches")
                            return True
                        else:
                            print("⚠️ Signed URL accessible but content doesn't match")
                            return False
                    else:
                        print(f"❌ Signed URL access failed: {access_response.status_code}")
                        return False
                else:
                    print("❌ No signed URL in response")
                    return False
            else:
                print(f"❌ Signed URL generation failed: {signed_url_response.status_code}")
                return False
        else:
            print(f"❌ NAS write failed: {write_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ NAS test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Task-6 Yotta Fallback Validation Test")
    print("=" * 50)

    # Test 1: Yotta routing
    yotta_success = test_yotta_fallback()

    # Test 2: NAS signed URLs
    nas_success = test_nas_signed_url()

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Yotta Routing: {'✅ PASS' if yotta_success else '❌ FAIL'}")
    print(f"NAS Signed URLs: {'✅ PASS' if nas_success else '❌ FAIL'}")

    if yotta_success and nas_success:
        print("\n🎉 ALL TESTS PASSED - Yotta fallback validation successful!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Check implementation")
        exit(1)