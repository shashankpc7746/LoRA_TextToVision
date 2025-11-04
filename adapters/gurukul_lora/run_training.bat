@echo off
REM Batch file to run training - may avoid PowerShell interruption issues
echo ============================================================
echo Gurukul LoRA Training Launcher (Batch Script)
echo ============================================================
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set TF_ENABLE_ONEDNN_OPTS=0

echo GPU: CUDA device 0
echo Memory: Expandable segments enabled
echo Dataset: datasets/gurukul_keyframes
echo.

REM Get epochs from command line or default to 100
if "%1"=="" (
    set EPOCHS=100
) else (
    set EPOCHS=%1
)

echo Training for %EPOCHS% epochs
echo.
echo Starting training... ^(this may take 2-3 hours^)
echo.

REM Run training
python adapters\gurukul_lora\train_adapter.py --dataset datasets\gurukul_keyframes --num_epochs %EPOCHS%

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS - Training completed!
    echo ============================================================
    echo.
    echo Adapter saved to: adapters\gurukul_lora.pt
) else (
    echo.
    echo ============================================================
    echo ERROR - Training failed with exit code: %ERRORLEVEL%
    echo ============================================================
)

pause
