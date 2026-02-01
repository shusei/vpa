<#
.SYNOPSIS
    執行專案的驗證檢查。
    
.DESCRIPTION
    執行 npm 測試套件，包含語法、標記、單元與完整性檢查。
    日誌將輸出至 .\artifacts\check\
    
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
    Write-Host "切換工作目錄至專案根目錄: $ProjectRoot"
    Set-Location $ProjectRoot
}

$LogDir = Join-Path $ProjectRoot "artifacts\check"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "check_$Timestamp.log"

Write-Host "開始執行檢查..."
Write-Host "日誌檔案: $LogFile"

function Invoke-Check {
    param (
        [string]$Name,
        [scriptblock]$ScriptBlock
    )
    
    Write-Host "[$Name] 執行中..."
    try {
        & $ScriptBlock | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "指令失敗，結束代碼 $LASTEXITCODE"
        }
        Write-Host "[$Name] 通過。" -ForegroundColor Green
    }
    catch {
        Write-Host "[$Name] 失敗。" -ForegroundColor Red
        Write-Error $_
    }
}

try {
    if (-not (Test-Path "node_modules")) {
        Write-Warning "找不到 node_modules。請先執行 'npm install' 或 '.\scripts\bootstrap.ps1'。"
        exit 1
    }

    Invoke-Check -Name "npm test" -ScriptBlock { npm test }

}
catch {
    Write-Host "檢查失敗！" -ForegroundColor Red
    $Error[0] | Out-File -FilePath $LogFile -Append -Encoding utf8
    exit 1
}
finally {
    $EndTime = Get-Date -Format "o"
    Write-Host "時間戳記: $EndTime" | Tee-Object -FilePath $LogFile -Append
    
    $Hash = Get-FileHash -Path $LogFile -Algorithm SHA256
    Write-Host "SHA256: $($Hash.Hash)"
    
    Write-Host "結束代碼: $LASTEXITCODE"
}
