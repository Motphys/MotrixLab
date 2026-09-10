# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

# Windows entry point: delegates to install.sh through the Git Bash bundled
# with Git for Windows. If the PowerShell execution policy blocks direct
# invocation, run: powershell -ExecutionPolicy Bypass -File install.ps1

Set-Location -Path $PSScriptRoot -ErrorAction Stop

# Prefer the Git Bash shipped with Git for Windows, derived from git.exe:
# ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
$bash = $null
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if ($git) {
    $candidate = $git -replace '\\cmd\\git\.exe$', '\bin\bash.exe' -replace '\\mingw64\\bin\\git\.exe$', '\bin\bash.exe'
    if ($candidate -ne $git -and (Test-Path $candidate)) {
        $bash = $candidate
    }
}

# Fall back to any bash on PATH (Git Bash, MSYS2, Cygwin or WSL all work).
if (-not $bash) {
    $bash = (Get-Command bash -ErrorAction SilentlyContinue).Source
}

if (-not $bash) {
    Write-Error "bash.exe not found: Git for Windows is required. Install it from https://git-scm.com/download/win and retry."
    exit 1
}

& $bash install.sh @args
exit $LASTEXITCODE
