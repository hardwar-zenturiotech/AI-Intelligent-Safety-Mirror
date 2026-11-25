# Creates a Windows Desktop shortcut for AI Intelligent Safety Mirror
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exePath = Join-Path $scriptDir "SafetyMirror.exe"
$iconCandidates = @(
    (Join-Path $scriptDir "resources\icon.ico"),
    (Join-Path $scriptDir "assets\icon.ico")
)
$iconPath = $iconCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopPath "AI Intelligent Safety Mirror.lnk"
$localShortcutPath = Join-Path $scriptDir "AI Intelligent Safety Mirror.lnk"

if (!(Test-Path $exePath)) {
    Write-Warning "SafetyMirror.exe not found. Running build_executable.ps1 first..."
    & (Join-Path $scriptDir "build_executable.ps1")
}

$wscript = New-Object -ComObject WScript.Shell

# Always create local shortcut in repository
$localShortcut = $wscript.CreateShortcut($localShortcutPath)
$localShortcut.TargetPath = $exePath
$localShortcut.WorkingDirectory = $scriptDir
$localShortcut.Description = "AI Intelligent Safety Mirror - Industrial PPE & SCBA Verification"
if (Test-Path $iconPath) {
    $localShortcut.IconLocation = "$iconPath,0"
}
$localShortcut.Save()
Write-Host "Created application shortcut at:`n  $localShortcutPath" -ForegroundColor Green

# Attempt to create Desktop shortcut
try {
    $desktopShortcut = $wscript.CreateShortcut($desktopShortcutPath)
    $desktopShortcut.TargetPath = $exePath
    $desktopShortcut.WorkingDirectory = $scriptDir
    $desktopShortcut.Description = "AI Intelligent Safety Mirror - Industrial PPE & SCBA Verification"
    if (Test-Path $iconPath) {
        $desktopShortcut.IconLocation = "$iconPath,0"
    }
    $desktopShortcut.Save()
    Write-Host "Created Desktop shortcut at:`n  $desktopShortcutPath" -ForegroundColor Green
} catch {
    Write-Host "Note: Could not place shortcut directly on Desktop in sandbox mode. The shortcut '$localShortcutPath' is ready to use or copy to Desktop." -ForegroundColor Yellow
}
