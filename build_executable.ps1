# Build Executable Script for AI Intelligent Safety Mirror
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Building AI Intelligent Safety Mirror Executable Software " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Build Frontend if needed
$distIndex = Join-Path $scriptDir "frontend\dist\index.html"
if (!(Test-Path $distIndex)) {
    Write-Host "[1/3] Building frontend production bundle..." -ForegroundColor Yellow
    Push-Location (Join-Path $scriptDir "frontend")
    npm run build
    Pop-Location
} else {
    Write-Host "[1/3] Production assets verified in frontend\dist." -ForegroundColor Green
}

# Step 2: Ensure Icon exists
$iconPath = Join-Path $scriptDir "assets\icon.ico"
if (!(Test-Path $iconPath)) {
    Write-Host "[2/3] Generating icon assets..." -ForegroundColor Yellow
    & ".\emosh_service\.venv\Scripts\python.exe" -c "
from PIL import Image, ImageDraw
s = 256
img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
draw.rounded_rectangle([8, 8, 248, 248], radius=48, fill=(13, 19, 31, 255), outline=(0, 210, 255, 255), width=10)
draw.ellipse([36, 36, 220, 220], outline=(255, 170, 0, 230), width=6)
draw.line([(70, 75), (186, 75)], fill=(0, 235, 255, 255), width=20)
draw.line([(186, 75), (70, 181)], fill=(0, 235, 255, 255), width=22)
draw.line([(70, 181), (186, 181)], fill=(0, 235, 255, 255), width=20)
draw.ellipse([175, 55, 195, 75], fill=(255, 170, 0, 255))
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save('assets/icon.ico', format='ICO', sizes=sizes)
"
} else {
    Write-Host "[2/3] Application icon verified at assets\icon.ico." -ForegroundColor Green
}

# Step 3: Compile C# Native Windows Executable
Write-Host "[3/3] Compiling native Windows executable (SafetyMirror.exe)..." -ForegroundColor Yellow
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$outExe = Join-Path $scriptDir "SafetyMirror.exe"
$manifest = Join-Path $scriptDir "launcher\SafetyMirror.manifest"
$source = Join-Path $scriptDir "launcher\Program.cs"

& $csc /nologo /target:winexe /optimize+ /out:$outExe /win32icon:$iconPath /win32manifest:$manifest /r:System.dll,System.Core.dll,System.Drawing.dll,System.Windows.Forms.dll $source

if (Test-Path $outExe) {
    $item = Get-Item $outExe
    $kb = [math]::Round($item.Length / 1KB, 1)
    Write-Host "`nBUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Executable generated: $outExe ($kb KB)" -ForegroundColor Green
    Write-Host "Double-click 'SafetyMirror.exe' to launch the desktop application." -ForegroundColor Cyan
} else {
    Write-Error "Failed to generate $outExe"
}
