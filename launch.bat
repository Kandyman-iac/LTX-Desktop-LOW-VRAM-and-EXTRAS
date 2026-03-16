@echo off
cd /d "%~dp0"
set ELECTRON_IS_DEV=0
npx electron .
