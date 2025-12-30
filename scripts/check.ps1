<#
.SYNOPSIS
    Runs validation checks for the project.
    
.DESCRIPTION
    Runs npm test suites including syntax, markup, unit, and integrity checks.
    Logs output to .\artifacts\check\
    
.EXAMPLE
    .\scripts\check.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ConfirmPreference = 'None'

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$CurrentLocation = Get-Location
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
if ($CurrentLocation.Path -ne $ProjectRoot.Path) {
    Write-Host "Changing location to ProjectRoot: $ProjectRoot"
    Set-Location $ProjectRoot
}

$LogDir = Join-Path $ProjectRoot "artifacts\check"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "check_$Timestamp.log"

Write-Host "Starting Check..."
Write-Host "Log File: $LogFile"

function Run-Check {
    param (
        [string]$Name,
        [scriptblock]$ScriptBlock
    )
    
    Write-Host "[$Name] Running..."
    try {
        & $ScriptBlock | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
        Write-Host "[$Name] Passed." -ForegroundColor Green
    }
    catch {
        Write-Host "[$Name] FAILED." -ForegroundColor Red
        Write-Error $_
    }
}

try {
    if (-not (Test-Path "node_modules")) {
        Write-Warning "node_modules not found. Please run 'npm install' or '.\scripts\bootstrap.ps1' first."
        exit 1
    }

    Write-Host "Running npm test..." | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    $TestCommand = { npm test }
    & $TestCommand | Tee-Object -FilePath $LogFile -Append
    
    if ($LASTEXITCODE -ne 0) {
        throw "npm test failed."
    }

}
catch {
    Write-Host "Check Failed!" -ForegroundColor Red
    $Error[0] | Out-File -FilePath $LogFile -Append -Encoding utf8
    exit 1
}
finally {
    $EndTime = Get-Date -Format "o"
    Write-Host "Timestamp: $EndTime" | Tee-Object -FilePath $LogFile -Append
    
    $Hash = Get-FileHash -Path $LogFile -Algorithm SHA256
    Write-Host "SHA256: $($Hash.Hash)"
    
    Write-Host "Exit Code: $LASTEXITCODE"
}
