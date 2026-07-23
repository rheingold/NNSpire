<#
.SYNOPSIS
    Full rebuild of NNSpire Agent — Web UI + Tauri desktop shell (Windows).

.DESCRIPTION
    This is a convenience wrapper that delegates to the official build script
    at nnagent/dev_helpers/build-desktop.ps1. It performs a complete release
    build:

      1. Build the Vite/React frontend (nnagent/web) → dist/
      2. Build the Tauri desktop binary (nnagent/desktop/src-tauri) — release mode

    Run this from any directory inside the repo.

.PARAMETER Clean
    Pass -Clean to perform `cargo clean` before building (full rebuild).

.PARAMETER Run
    Pass -Run to launch nnagent.exe after a successful build.

.EXAMPLE
    # Quick rebuild (incremental)
    .\docs\commands\nnagent_compile_all_win_Tauri.ps1

    # Full clean rebuild
    .\docs\commands\nnagent_compile_all_win_Tauri.ps1 -Clean

    # Rebuild and launch
    .\docs\commands\nnagent_compile_all_win_Tauri.ps1 -Clean -Run
#>

param(
    [switch]$Clean,
    [switch]$Run
)

# Resolve paths - script is at docs/commands/, repo root is 2 levels up
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot    = Split-Path -Parent (Split-Path -Parent $ScriptDir)
# The build script lives inside nnagent/ subdirectory
$NnagentDir = Join-Path $RepoRoot 'nnagent'
$BuildScript = Join-Path $NnagentDir 'dev_helpers\build-desktop.ps1'

if (-not (Test-Path $BuildScript)) {
    Write-Host "ERROR: Build script not found at $BuildScript" -ForegroundColor Red
    exit 1
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  NNSpire Agent - Full Compile (Windows + Tauri)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Repo root    : $RepoRoot" -ForegroundColor DarkGray
Write-Host "Build script : $BuildScript" -ForegroundColor DarkGray
Write-Host ""

# Forward arguments to the build script
$ArgsList = @()
if ($Clean) { $ArgsList += '-Clean' }
if ($Run)   { $ArgsList += '-Run'   }

& $BuildScript @ArgsList
exit $LASTEXITCODE