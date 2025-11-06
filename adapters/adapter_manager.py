"""
LoRA Adapter Manager for Task-7 Quality Leap
Manages multiple LoRA adapters and their application
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from .lora_adapter import LoRAAdapter, GurukulLoRA


class AdapterManager:
    """Manages multiple LoRA adapters"""

    def __init__(self, adapters_dir: str = "adapters"):
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(exist_ok=True)

        self.adapters: Dict[str, LoRAAdapter] = {}
        self.metadata_file = self.adapters_dir / "adapters_metadata.json"

        # Load existing adapters
        self._load_metadata()

    def _load_metadata(self):
        """Load adapter metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded metadata for {len(metadata)} adapters")
            except Exception as e:
                print(f"Warning: Could not load adapter metadata: {e}")

    def _save_metadata(self):
        """Save adapter metadata"""
        metadata = {}
        for name, adapter in self.adapters.items():
            metadata[name] = {
                "path": str(adapter.adapter_path),
                "base_model": adapter.base_model_path,
                "created": datetime.now().isoformat(),
                "is_loaded": adapter.is_loaded
            }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def register_adapter(self, name: str, adapter: LoRAAdapter):
        """Register a new adapter"""
        self.adapters[name] = adapter
        self._save_metadata()
        print(f"Registered adapter: {name}")

    def get_adapter(self, name: str) -> Optional[LoRAAdapter]:
        """Get adapter by name"""
        return self.adapters.get(name)

    def list_adapters(self) -> List[str]:
        """List all registered adapters"""
        return list(self.adapters.keys())

    def load_gurukul_adapter(self) -> GurukulLoRA:
        """Load or create Gurukul LoRA adapter"""
        # =================================================================
        # TASK 10: Verify Adapter Signature Before Loading
        # =================================================================
        print("\n🔒 Verifying adapter signature...")
        
        try:
            import sys
            import os
            from pathlib import Path
            
            # Add parent directory to path for security imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from security.artifact_signer import ArtifactSigner
            
            # Path to gurukul_lora checkpoint
            adapter_path = Path("adapters/gurukul_lora")
            checkpoint_paths = list(adapter_path.glob("*.pt")) + list(adapter_path.glob("*.safetensors"))
            
            if checkpoint_paths:
                checkpoint_file = checkpoint_paths[0]  # Use first found checkpoint
                signature_file = Path(str(checkpoint_file) + '.sig')
                
                # Check if signature exists
                if signature_file.exists():
                    print(f"   📝 Found checkpoint: {checkpoint_file.name}")
                    print(f"   🔏 Found signature: {signature_file.name}")
                    
                    # Get public key for verification
                    public_key_path = os.getenv('ARTIFACT_PUBLIC_KEY_PATH', 'security/keys/signing_key.pub')
                    
                    if Path(public_key_path).exists():
                        # Verify signature
                        signer = ArtifactSigner(public_key_path)
                        is_valid = signer.verify_signature(str(checkpoint_file))
                        
                        if is_valid:
                            print(f"   ✅ Signature verified successfully")
                        else:
                            print(f"   ❌ SIGNATURE VERIFICATION FAILED")
                            print(f"   🚨 Unsigned or tampered model detected!")
                            
                            # Check if in production mode (via environment)
                            runtime_mode = os.getenv('RUNTIME_MODE', 'production')
                            
                            if runtime_mode == 'production':
                                raise ValueError(
                                    "SECURITY VIOLATION: Cannot load unsigned model in production mode. "
                                    "Sign the model using: python -m security.artifact_signer sign adapters/gurukul_lora/*.pt"
                                )
                            else:
                                print(f"   ⚠️  WARNING: Loading unsigned model in development mode")
                                print(f"   📝 Sign the model before production deployment")
                    else:
                        print(f"   ⚠️  Public key not found at: {public_key_path}")
                        print(f"   📝 Skipping signature verification")
                else:
                    print(f"   ⚠️  No signature file found for: {checkpoint_file.name}")
                    print(f"   📝 Expected: {signature_file.name}")
                    print(f"   💡 Sign using: python tools/sign_artifact.py {checkpoint_file}")
                    
                    # In production, this should fail
                    runtime_mode = os.getenv('RUNTIME_MODE', 'development')
                    if runtime_mode == 'production':
                        raise ValueError(
                            "SECURITY VIOLATION: Cannot load unsigned model in production mode. "
                            f"Sign the model using: python tools/sign_artifact.py {checkpoint_file}"
                        )
            else:
                print(f"   ℹ️  No checkpoint files found in {adapter_path}")
        
        except ImportError as e:
            print(f"   ⚠️  Security module not available: {e}")
            print(f"   📝 Loading adapter without verification (development mode)")
        except Exception as e:
            print(f"   ⚠️  Signature verification error: {e}")
            # Don't fail the load, just warn
            print(f"   📝 Continuing with adapter load...")
        
        print("="*70 + "\n")
        # =================================================================
        # END: Signature Verification
        # =================================================================
        
        gurukul_lora = GurukulLoRA()

        if not gurukul_lora.is_trained():
            print("Gurukul LoRA adapter not trained yet")
            print("Use adapter_trainer.py to train the adapter first")
        else:
            print("Gurukul LoRA adapter loaded successfully")

        return gurukul_lora

    def create_adapter_from_config(self, name: str, config: Dict[str, Any]) -> LoRAAdapter:
        """Create adapter from configuration"""
        adapter = LoRAAdapter(
            base_model_path=config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        )

        # Apply custom LoRA config if provided
        if "lora_config" in config:
            lora_config = config["lora_config"]
            adapter.lora_config.r = lora_config.get("r", 16)
            adapter.lora_config.lora_alpha = lora_config.get("lora_alpha", 32)
            adapter.lora_config.lora_dropout = lora_config.get("lora_dropout", 0.1)

        self.register_adapter(name, adapter)
        return adapter

    def benchmark_adapters(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark all adapters on test prompts"""
        results = {}

        for name, adapter in self.adapters.items():
            print(f"Benchmarking adapter: {name}")

            try:
                # Load adapter
                pipeline = adapter.apply_lora_adapter()

                # Test generation
                test_results = []
                for prompt in test_prompts[:3]:  # Test first 3 prompts
                    result = adapter.generate_with_adapter(prompt, num_inference_steps=20)
                    test_results.append({
                        "prompt": prompt,
                        "success": len(result.get("images", [])) > 0,
                        "num_images": len(result.get("images", []))
                    })

                success_rate = sum(1 for r in test_results if r["success"]) / len(test_results)

                results[name] = {
                    "success_rate": success_rate,
                    "test_results": test_results,
                    "status": "success"
                }

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def cleanup_unused_adapters(self):
        """Clean up adapters that are no longer registered"""
        # This would check for adapter files that aren't in metadata
        # and remove them to save disk space
        print("Adapter cleanup not implemented yet")

    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get statistics about registered adapters"""
        total_adapters = len(self.adapters)
        loaded_adapters = sum(1 for a in self.adapters.values() if a.is_loaded)
        trained_adapters = sum(1 for a in self.adapters.values() if a.adapter_path.exists())

        return {
            "total_adapters": total_adapters,
            "loaded_adapters": loaded_adapters,
            "trained_adapters": trained_adapters,
            "adapters": list(self.adapters.keys())
        }


# Global adapter manager instance
_adapter_manager = None


def get_adapter_manager() -> AdapterManager:
    """Get global adapter manager instance"""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager


def quick_setup_gurukul_adapter():
    """Quick setup for Gurukul LoRA adapter"""
    manager = get_adapter_manager()

    # Create Gurukul adapter
    gurukul_adapter = LoRAAdapter()
    manager.register_adapter("gurukul_lora", gurukul_adapter)

    # Create Gurukul LoRA wrapper
    gurukul_lora = GurukulLoRA()

    print("Gurukul LoRA adapter setup complete")
    print("Use adapter_trainer.py to train the adapter on keyframes")

    return gurukul_lora


# Utility functions
def list_available_adapters():
    """List all available adapters"""
    manager = get_adapter_manager()
    adapters = manager.list_adapters()

    if not adapters:
        print("No adapters registered")
        return

    print("Available adapters:")
    for adapter_name in adapters:
        adapter = manager.get_adapter(adapter_name)
        status = "loaded" if adapter and adapter.is_loaded else "not loaded"
        trained = "trained" if adapter and adapter.adapter_path.exists() else "not trained"
        print(f"  - {adapter_name}: {status}, {trained}")


def test_adapter_generation(adapter_name: str, prompt: str):
    """Test generation with specific adapter"""
    manager = get_adapter_manager()
    adapter = manager.get_adapter(adapter_name)

    if not adapter:
        print(f"Adapter '{adapter_name}' not found")
        return

    print(f"Testing adapter: {adapter_name}")
    print(f"Prompt: {prompt}")

    try:
        result = adapter.generate_with_adapter(prompt)
        print(f"Generation successful: {len(result.get('images', []))} images created")
    except Exception as e:
        print(f"Generation failed: {e}")