"""
Clear GPU memory before training
Run this if you get OOM errors
"""

import torch
import gc

print("Clearing GPU memory...")

if torch.cuda.is_available():
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Force garbage collection
    gc.collect()
    
    # Check memory status
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU memory cleared!")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved:  {reserved:.2f} GB")
    print(f"   Total:     {total:.2f} GB")
    print(f"   Free:      {total - allocated:.2f} GB")
else:
    print("❌ CUDA not available")
