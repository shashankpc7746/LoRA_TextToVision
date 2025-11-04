# PowerShell script to run training with proper environment setup
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Gurukul LoRA Training Launcher" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TF_ENABLE_ONEDNN_OPTS = "0"  # Silence TensorFlow warnings

Write-Host "✅ GPU: CUDA device 0" -ForegroundColor Green
Write-Host "✅ Memory: Expandable segments enabled" -ForegroundColor Green
Write-Host "✅ Dataset: datasets/gurukul_keyframes" -ForegroundColor Green
Write-Host ""

# Get epochs from command line or default to 100
$epochs = if ($args.Count -gt 0) { $args[0] } else { "100" }
Write-Host "🎯 Training for $epochs epochs" -ForegroundColor Cyan
Write-Host ""

# Run training
Write-Host "Starting training..." -ForegroundColor Yellow
python adapters/gurukul_lora/train_adapter.py --dataset datasets/gurukul_keyframes --num_epochs $epochs

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "SUCCESS - Training completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Adapter saved to: adapters/gurukul_lora.pt" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "ERROR - Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
}
