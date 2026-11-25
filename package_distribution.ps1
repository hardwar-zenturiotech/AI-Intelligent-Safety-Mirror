# Packages AI Intelligent Safety Mirror into a standalone, distributable folder
# All supporting files are consolidated into a single "resources" folder alongside SafetyMirror.exe
param(
    [string]$OutputDirName = "AI-Safety-Mirror-Portable",
    [switch]$Zip
)
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$outputDirName = $OutputDirName
$targetPackage = Join-Path $scriptDir $outputDirName
$resourcesDir = Join-Path $targetPackage "resources"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Packaging AI Intelligent Safety Mirror Distribution     " -ForegroundColor Cyan
Write-Host " Destination: $outputDirName                             " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Rebuild Frontend with Latest Source
$frontendDist = Join-Path $scriptDir "frontend\dist"
Write-Host "[1/4] Compiling latest production web assets (npm run build)..." -ForegroundColor Yellow
Push-Location (Join-Path $scriptDir "frontend")
npm run build
Pop-Location

# Step 2: Compile / Verify SafetyMirror.exe
Write-Host "[2/4] Verifying and compiling SafetyMirror.exe..." -ForegroundColor Yellow
& (Join-Path $scriptDir "build_executable.ps1")

# Step 3: Clean & Prepare Distribution Folder
Write-Host "[3/4] Assembling standalone distribution package in $outputDirName..." -ForegroundColor Yellow
if (Test-Path $targetPackage) {
    Remove-Item -Path $targetPackage -Recurse -Force
}
New-Item -Path $targetPackage -ItemType Directory -Force | Out-Null
New-Item -Path $resourcesDir -ItemType Directory -Force | Out-Null

# Copy Primary Executable & Launchers into package root
Copy-Item (Join-Path $scriptDir "SafetyMirror.exe") -Destination $targetPackage -Force
Copy-Item (Join-Path $scriptDir "Start-Safety-Mirror.bat") -Destination $targetPackage -Force
Copy-Item (Join-Path $scriptDir "CreateDesktopShortcut.bat") -Destination $targetPackage -Force
Copy-Item (Join-Path $scriptDir "Install-Desktop-Shortcut.ps1") -Destination $targetPackage -Force

# Step 4: Copy ALL Supporting Files into the single "resources" folder
Write-Host "[4/4] Consolidating all supporting files into 'resources\'..." -ForegroundColor Yellow

# Copy web bundle (index.html, assets, models, mediapipe wasm)
Copy-Item -Path "$frontendDist\*" -Destination $resourcesDir -Recurse -Force

# Copy icon
if (Test-Path (Join-Path $scriptDir "assets\icon.ico")) {
    Copy-Item (Join-Path $scriptDir "assets\icon.ico") -Destination $resourcesDir -Force
}

# Copy server.mjs
if (Test-Path (Join-Path $scriptDir "server.mjs")) {
    Copy-Item (Join-Path $scriptDir "server.mjs") -Destination $resourcesDir -Force
}

# Copy EMOSH microservice scripts (excluding venv) into resources/emosh_service
$emoshDest = Join-Path $resourcesDir "emosh_service"
New-Item -Path $emoshDest -ItemType Directory -Force | Out-Null
$emoshSource = Join-Path $scriptDir "emosh_service"
Get-ChildItem -Path $emoshSource -File | ForEach-Object {
    Copy-Item $_.FullName -Destination $emoshDest -Force
}

# Generate clear portable README.txt in package root
$readmeContent = @"
================================================================================
           AI INTELLIGENT SAFETY MIRROR (PORTABLE EDITION)
================================================================================

HOW TO RUN:
1. Double-click "SafetyMirror.exe" (or "Start-Safety-Mirror.bat").
2. The application opens in a dedicated, hardware-accelerated desktop window.
3. Allow camera permissions when prompted (permissions are saved automatically).

OPTIONAL DESKTOP SHORTCUT:
- Double-click "CreateDesktopShortcut.bat" to place a shortcut on your
  Windows Desktop.

FOLDER CONTENTS:
- SafetyMirror.exe          : Native Windows GUI application launcher & HTTP server
- Start-Safety-Mirror.bat   : Batch file launcher
- CreateDesktopShortcut.bat : Installer for Desktop shortcut
- resources/                : CONTAINS ALL SUPPORTING FILES:
                              - 3D Avatar models (avatar.glb, scba_mask.fbx)
                              - Computer-vision neural models (RTMW3D, RTMPose, MediaPipe)
                              - TensorFlow BodyPix semantic segmentation engine
                              - WebAssembly high-performance runtime
                              - Compiled web interface assets (HTML, CSS, JS)
                              - EMOSH 3D human body reconstruction viewer assets

SYSTEM REQUIREMENTS:
- Windows 10 or Windows 11 (64-bit)
- Google Chrome or Microsoft Edge (pre-installed on Windows)
- Standard USB or built-in webcam
- GPU hardware acceleration (Direct3D 11, Vulkan, WebGPU) supported

RECORDINGS:
- When you click "Record clip" in the mirror, saved videos are automatically
  stored in a "recordings\" folder created right alongside SafetyMirror.exe.

================================================================================
AI Intelligent Safety Mirror System - Industrial PPE Verification
================================================================================
"@

Set-Content -Path (Join-Path $targetPackage "README.txt") -Value $readmeContent -Encoding UTF8

# Measure total size
$totalBytes = (Get-ChildItem -Path $targetPackage -Recurse | Measure-Object -Property Length -Sum).Sum
$totalMB = [math]::Round($totalBytes / 1MB, 1)
$fileCount = (Get-ChildItem -Path $targetPackage -Recurse -File | Measure-Object).Count

Write-Host "`nPACKAGE COMPLETE!" -ForegroundColor Green
Write-Host "Target Folder:  $targetPackage" -ForegroundColor Cyan
Write-Host "Total Size:     $totalMB MB ($fileCount files)" -ForegroundColor Cyan
Write-Host "Structure:" -ForegroundColor Yellow
Write-Host "  $outputDirName\" -ForegroundColor White
Write-Host "  |-- SafetyMirror.exe          (Main Executable)" -ForegroundColor White
Write-Host "  |-- Start-Safety-Mirror.bat   (Batch Launcher)" -ForegroundColor White
Write-Host "  |-- CreateDesktopShortcut.bat (Desktop Shortcut Creator)" -ForegroundColor White
Write-Host "  |-- README.txt                (User Documentation)" -ForegroundColor White
Write-Host "  `-- resources\                (ALL Supporting Models, WASM, Web Assets)" -ForegroundColor White
Write-Host "`nYou can now share the '$outputDirName' folder (or zip it) with anyone!" -ForegroundColor Green

if ($Zip) {
    $zipPath = Join-Path $scriptDir "$outputDirName.zip"
    Write-Host "`nCreating ZIP archive: $zipPath..." -ForegroundColor Yellow
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($targetPackage, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $zipMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host "ZIP archive created: $zipPath ($zipMB MB)" -ForegroundColor Green
}
