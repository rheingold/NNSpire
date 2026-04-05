#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fully automated markdown export using local Pandoc installation.
    Zero manual interaction required.

.EXAMPLE
    .\export-final.ps1
#>

$StartTime = Get-Date
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StandardsDir = Join-Path $RootDir 'standards'
$AnnexesDir = Join-Path $RootDir 'annexes'
$PdfDir = Join-Path $RootDir 'pdf'
$PandocExe = "C:\Users\plachy\Documents\Dev\_ToolsPanDoc\pandoc.exe"
$WkHtmlToPdf = "C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

if (-not (Test-Path $PdfDir)) {
    New-Item -ItemType Directory -Path $PdfDir -Force | Out-Null
}

Write-Host "==============================================="
Write-Host "FULLY AUTOMATED PDF EXPORT"
Write-Host "==============================================="
Write-Host ""

# Verify tools
if (-not (Test-Path $PandocExe)) {
    Write-Host "ERROR: Pandoc not found at $PandocExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $WkHtmlToPdf)) {
    Write-Host "ERROR: wkhtmltopdf not found at $WkHtmlToPdf" -ForegroundColor Red
    exit 1
}

Write-Host "Using: Pandoc + wkhtmltopdf"
Write-Host "Output: $PdfDir"
Write-Host ""

# Collect files
$mdFiles = @()
$mdFiles += Get-ChildItem -Path $StandardsDir -Filter '*.md' -ErrorAction SilentlyContinue
$mdFiles += Get-ChildItem -Path $AnnexesDir -Filter '*.md' -ErrorAction SilentlyContinue
$rootFiles = @(Get-ChildItem -Path $RootDir -Filter '*.md' -ErrorAction SilentlyContinue | 
    Where-Object { -not $_.PSIsContainer -and $_.Name -notlike 'export*' -and $_.Name -notlike 'verify*' })
$mdFiles += $rootFiles

Write-Host "Found $($mdFiles.Count) markdown files"
Write-Host ""

$success = 0
$fail = 0

foreach ($file in $mdFiles) {
    $outputPdf = Join-Path $PdfDir "$($file.BaseName).pdf"
    
    Write-Host "Converting: $($file.Name)..." -NoNewline
    
    try {
        # Create temp files
        $tempHtml = Join-Path $env:TEMP "$($file.BaseName).html"
        $tempCss = Join-Path $env:TEMP "narrow-margins.css"
        
        # Create CSS to override Pandoc's default padding/width and style code blocks
        @"
body {
    margin: 0 !important;
    padding: 20px !important;
    max-width: 100% !important;
}

pre {
    background-color: #f5f5f5 !important;
    border: 1px solid #ddd !important;
    border-radius: 4px !important;
    padding: 15px !important;
    margin: 15px 0 !important;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 12px !important;
    overflow-x: auto !important;
    line-height: 1.5 !important;
    letter-spacing: 0.5px !important;
}

pre code {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
}

code {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    background-color: #f0f0f0 !important;
    padding: 2px 4px !important;
    border-radius: 2px !important;
}
"@ | Out-File -FilePath $tempCss -Encoding UTF8
        
        # Convert MD to HTML with Pandoc (need to be in same directory for relative paths)
        $originalDir = Get-Location
        try {
            Set-Location (Split-Path $file.FullName)
            & "$PandocExe" (Split-Path $file.FullName -Leaf) `
                -o "$tempHtml" `
                --standalone `
                --embed-resources `
                --number-sections `
                --katex `
                --css="$tempCss" `
                --metadata title="$($file.BaseName)" `
                2>$null
        } finally {
            Set-Location $originalDir
        }
        
        # Step 2: HTML to PDF with narrow margins
        & "$WkHtmlToPdf" `
            --margin-top 10mm `
            --margin-bottom 10mm `
            --margin-left 10mm `
            --margin-right 10mm `
            --enable-local-file-access `
            "$tempHtml" `
            "$outputPdf" `
            2>$null
        
        # Cleanup
        Remove-Item $tempHtml -Force -ErrorAction SilentlyContinue
        Remove-Item $tempCss -Force -ErrorAction SilentlyContinue
        
        if (Test-Path $outputPdf) {
            $size = [math]::Round((Get-Item $outputPdf).Length / 1KB, 2)
            Write-Host " OK ($size KB)" -ForegroundColor Green
            $success++
        } else {
            Write-Host " FAILED" -ForegroundColor Red
            $fail++
        }
    }
    catch {
        Write-Host " ERROR" -ForegroundColor Red
        $fail++
    }
}

$elapsed = (Get-Date) - $StartTime

Write-Host ""
Write-Host "==============================================="
Write-Host "COMPLETE"
Write-Host "==============================================="
Write-Host "Success: $success files" -ForegroundColor Green
Write-Host "Failed: $fail files" -ForegroundColor $(if ($fail -gt 0) { "Red" } else { "Green" })
Write-Host "Time: $([math]::Round($elapsed.TotalSeconds))s"
Write-Host "Location: $PdfDir"
Write-Host "==============================================="
