@echo off
REM Quick Training Status Monitor
REM Run this anytime to check training progress

:LOOP
cls
echo ============================================================
echo TRAINING STATUS MONITOR
echo ============================================================
echo.
echo Current Time: %TIME%
echo.

REM Check if adapter file exists
if exist "adapters\gurukul_lora.pt" (
    echo [92mSTATUS: TRAINING COMPLETE![0m
    echo.
    for %%F in ("adapters\gurukul_lora.pt") do (
        echo Adapter File: %%~nxF
        echo Size: %%~zF bytes
        echo Modified: %%~tF
    )
    echo.
    echo [92mYou can now use the adapter for inference![0m
    echo.
    pause
    exit
) else (
    echo [93mSTATUS: TRAINING IN PROGRESS...[0m
    echo.
    echo Adapter file not created yet
)

echo.
echo Checking for checkpoints...
if exist "adapters\gurukul_lora\checkpoint_epoch_25.pt" (
    echo   [92m+ Checkpoint 25[0m - Completed
) else (
    echo   [90m- Checkpoint 25[0m - Not yet
)

if exist "adapters\gurukul_lora\checkpoint_epoch_50.pt" (
    echo   [92m+ Checkpoint 50[0m - Completed
) else (
    echo   [90m- Checkpoint 50[0m - Not yet
)

if exist "adapters\gurukul_lora\checkpoint_epoch_75.pt" (
    echo   [92m+ Checkpoint 75[0m - Completed
) else (
    echo   [90m- Checkpoint 75[0m - Not yet
)

echo.
echo ============================================================
echo.
echo [36mPress Ctrl+C to exit, or wait 30 seconds for auto-refresh...[0m
timeout /t 30 /nobreak >nul
goto LOOP
