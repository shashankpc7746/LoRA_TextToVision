@echo off
REM ============================================================
REM Gurukul LoRA Training - Run in Fresh Terminal
REM ============================================================

echo.
echo ============================================================
echo Gurukul LoRA Training
echo ============================================================
echo.
echo This will train the LoRA adapter for 100 epochs (~2-3 hours)
echo Please DO NOT close this window or press any keys during training
echo.
echo IMPORTANT: Just started the training script with fixes for SDXL!
echo            Model will load, then training will begin.
echo.
pause

REM Change to project directory
cd /d C:\Shashank\LoRA_TextToVision

REM Activate virtual environment
echo Activating gurukul-lora-env...
call gurukul-lora-env\Scripts\activate.bat

REM Verify environment
echo.
echo Checking environment...
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================================
echo Starting Training - This will take 2-3 hours
echo DO NOT CLOSE THIS WINDOW
echo ============================================================
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set TF_ENABLE_ONEDNN_OPTS=0

REM Run training
python adapters\gurukul_lora\train_adapter.py --dataset datasets\gurukul_keyframes --num_epochs 100

REM Check result
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Training completed successfully!
    echo ============================================================
    echo.
    echo Adapter saved to: adapters\gurukul_lora.pt
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Training failed with exit code %ERRORLEVEL%
    echo ============================================================
    echo.
)

pause
