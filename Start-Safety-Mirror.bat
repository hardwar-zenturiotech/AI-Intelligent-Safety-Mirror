@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0SafetyMirror.exe" (
    echo SafetyMirror.exe not found. Building application executable...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_executable.ps1"
)

start "" "%~dp0SafetyMirror.exe" %*
