<#
.SYNOPSIS
    Build the NNSpire Agent desktop application (Tauri + Vite frontend).

.DESCRIPTION
    This script performs a full release build of the NNSpire Agent desktop app:
      1. Builds the Vite frontend (nnagent/web) into dist/
      2. Builds the Tauri desktop binary (nnagent/desktop/src-tauri) in release mode

    The portable Rust toolchain at _ToolRust is used automatically.

.PARAMETER Clean
        If set, performs cargo clean before building.

.PARAMETER Run
        If set, launches the built nnagent.exe after a successful build.

.EXAMPLE
    .\dev_helpers\build-desktop.ps1
    .\dev_helpers\build-desktop.ps1 -Clean
    .\dev_helpers\build-desktop.ps1 -Run
    .\dev_helpers\build-desktop.ps1 -Clean -Run
#>

param(
    [switch]$Clean,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'

# ── Paths ──────────────────────────────────────────────────────────────────────
$RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$WebDir      = Join-Path $RepoRoot 'web'
$TauriDir    = Join-Path $RepoRoot 'desktop\src-tauri'
$RustRoot    = 'D:\plachy\Dokumenty\Dev\_ToolRust'
$TargetDir   = Join-Path $RepoRoot 'desktop\target'
$ExePath     = Join-Path $TargetDir 'release\nnagent.exe'

# ── Environment ────────────────────────────────────────────────────────────────
$env:RUSTUP_HOME   = Join-Path $RustRoot 'rustup'
$env:CARGO_HOME    = Join-Path $RustRoot 'cargo'
$env:PATH          = "$(Join-Path $RustRoot 'cargo\bin');$env:PATH"
$env:CARGO_TARGET_DIR = $TargetDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NNSpire Agent - Desktop Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Frontend ──────────────────────────────────────────────────────────
Write-Host "[1/3] Building frontend (Vite) ..." -ForegroundColor Yellow
Push-Location $WebDir
try {
    $result = pnpm build 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Frontend build failed." -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        exit 1
    }
    Write-Host "  Frontend built successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}

# ── Step 2: Optional Clean ────────────────────────────────────────────────────
if ($Clean) {
    Write-Host ""
    Write-Host "[2/3] Cleaning Tauri target directory ..." -ForegroundColor Yellow
    Push-Location $TauriDir
    try {
        cargo clean 2>&1 | Out-Null
        Write-Host "  Cleaned." -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# ── Step 3: Tauri Build ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/3] Building Tauri desktop (release) ..." -ForegroundColor Yellow
Push-Location $TauriDir
try {
    $result = cargo build --release 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Tauri build failed." -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        exit 1
    }
    Write-Host "  Tauri build succeeded." -ForegroundColor Green
}
finally {
    Pop-Location
}

# ── Verify ─────────────────────────────────────────────────────────────────────
Write-Host ""
if (Test-Path $ExePath) {
    $size = [math]::Round((Get-Item $ExePath).Length / 1MB, 2)
    Write-Host "[OK] nnagent.exe built - ${size} MB" -ForegroundColor Green
    Write-Host "     Path: $ExePath" -ForegroundColor DarkGray
}
else {
    Write-Host "ERROR: Expected binary not found at $ExePath" -ForegroundColor Red
    exit 1
}

# ── Step 4: Optional Launch ───────────────────────────────────────────────────
if ($Run) {
    Write-Host ""
    Write-Host "Launching nnagent.exe ..." -ForegroundColor Yellow
    Start-Process $ExePath
    Write-Host "  Launched." -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build complete." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
