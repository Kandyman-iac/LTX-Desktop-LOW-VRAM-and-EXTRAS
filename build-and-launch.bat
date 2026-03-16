@echo off
cd /d "%~dp0"
echo Building frontend...
call pnpm run build:frontend
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)
echo Launching...
set ELECTRON_IS_DEV=0
npx electron .
