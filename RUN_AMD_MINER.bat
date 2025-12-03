@echo off
echo ============================================
echo BTCW AMD Miner Launcher
echo ============================================
echo.

if not exist "miner_amd.exe" (
    echo ERROR: miner_amd.exe not found!
    echo Please compile first using build instructions.
    pause
    exit /b 1
)

echo Starting BTCW AMD Miner...
echo Make sure bitcoin-pow-qt.exe is running!
echo.

miner_amd.exe 0

pause
