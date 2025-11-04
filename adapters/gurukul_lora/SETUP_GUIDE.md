# Task 9 Day 1: Quick Setup Guide
## Indigenous Gurukul LoRA Adapter

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (RTX 3080 recommended)
- 16GB+ VRAM
- Virtual environment activated: `gurukul-lora-env`

### Step 1: Install Additional Dependencies

```bash
pip install peft accelerate bitsandbytes
```

### Step 2: Prepare Dataset

Option A: Use Placeholder Dataset (for testing)
```bash
python adapters/gurukul_lora/dataset_curator.py --create_placeholder --num_images 100
```

Option B: Use Curated Dataset (for production)
```bash
# Place your 50-200 curated keyframes in datasets/gurukul_keyframes/
# Then validate:
python adapters/gurukul_lora/dataset_curator.py --validate_only
```

### Step 3: Run Quick Setup Test

```bash
python adapters/gurukul_lora/test_adapter.py --quick_setup
```

This will:
- ✅ Create placeholder dataset (if needed)
- ✅ Validate dataset structure
- ✅ Check CUDA availability
- ✅ Verify all components are ready

### Step 4: Train Adapter

Quick Training (10 epochs, ~15-20 minutes on RTX 3080):
```bash
python adapters/gurukul_lora/train_adapter.py \
  --dataset datasets/gurukul_keyframes \
  --num_epochs 10 \
  --output_dir adapters/gurukul_lora
```

Full Training (100 epochs, ~2-3 hours on RTX 3080):
```bash
python adapters/gurukul_lora/train_adapter.py \
  --dataset datasets/gurukul_keyframes \
  --num_epochs 100 \
  --output_dir adapters/gurukul_lora
```

### Step 5: Verify Training

```bash
# Check adapter file exists
ls -lh adapters/gurukul_lora/gurukul_lora.pt

# Check metadata exists
cat adapters/gurukul_lora/metadata.json
```

Expected output:
```
gurukul_lora.pt       # Trained adapter (~50-100MB)
metadata.json         # Training metadata with deterministic config
checkpoint_*.pt       # Training checkpoints
```

### Step 6: Test Generation

Single Generation:
```bash
python adapters/gurukul_lora/inference.py \
  --prompt "Ancient Gurukul classroom with teacher and students" \
  --seed 42 \
  --output_dir outputs/indigenous_keyframes
```

Verify Determinism:
```bash
python adapters/gurukul_lora/inference.py \
  --prompt "Traditional Vedic learning under banyan tree" \
  --seed 42 \
  --verify_determinism
```

Batch Generation:
```bash
python adapters/gurukul_lora/inference.py \
  --prompt "Students practicing yoga in Gurukul" \
  --seed 42 \
  --batch \
  --num_variations 5
```

### Step 7: Run Full Test Suite

```bash
python adapters/gurukul_lora/test_adapter.py --full_suite
```

This will test:
- ✅ Dataset curator
- ✅ Dataset validation
- ✅ Adapter existence
- ✅ Metadata completeness
- ✅ Generator loading
- ✅ Deterministic generation

### Expected Timeline

| Task | Duration | GPU |
|------|----------|-----|
| Dataset Preparation | 5-10 min | N/A |
| Quick Setup Test | 2-3 min | N/A |
| Quick Training (10 epochs) | 15-20 min | RTX 3080 |
| Full Training (100 epochs) | 2-3 hours | RTX 3080 |
| Generation Test | 1-2 min | RTX 3080 |
| Full Test Suite | 5-10 min | RTX 3080 |

### Troubleshooting

**Issue: CUDA Out of Memory**
```bash
# Reduce batch size
python adapters/gurukul_lora/train_adapter.py \
  --dataset datasets/gurukul_keyframes \
  --num_epochs 10 \
  --batch_size 1
```

**Issue: Adapter not loading**
```bash
# Check file exists and is valid
python -c "import torch; print(torch.load('adapters/gurukul_lora/gurukul_lora.pt').keys())"
```

**Issue: Generation fails**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Try CPU generation (slower)
python adapters/gurukul_lora/inference.py \
  --prompt "Test prompt" \
  --device cpu
```

### Acceptance Criteria (Day 1)

✅ **Required Deliverables:**
- [ ] `adapters/gurukul_lora/gurukul_lora.pt` exists
- [ ] `adapters/gurukul_lora/metadata.json` exists with:
  - deterministic_config (seed, cfg_scale, scheduler)
  - ksml_lineage (parent_models, training_dataset, adapter_type)
  - model_hash for lineage tracking
- [ ] Deterministic generation verified (same prompt + seed = same output)
- [ ] Metadata stored to NAS location

✅ **Quality Checks:**
- [ ] Adapter size: 50-200 MB (LoRA should be lightweight)
- [ ] Generation time: < 30 seconds per keyframe on RTX 3080
- [ ] Determinism test: 100% identical outputs for same seed
- [ ] Metadata completeness: All required fields present

### Next Steps (Day 2)

Once Day 1 is complete:
1. Implement `interpolator/temporal_consistency.py`
2. Create temporal UNet denoiser
3. Implement de-flicker pass
4. Expose `process_frames_consistent()` API

---

**Day 1 Status**: 🚀 READY TO TRAIN

For questions or issues, refer to:
- Task-9-README.md (comprehensive documentation)
- test_adapter.py (validation tests)
