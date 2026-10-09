#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Register KHQuant dashboard autostart and daily update scheduled tasks on Windows.
.DESCRIPTION
    Creates two scheduled tasks:
      - KHQuant Dashboard : runs at user logon
      - KHQuant Daily Update : runs daily at 09:00
    Run from an elevated PowerShell prompt.
#>
param(
    [string]$RepoRoot = "",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
}

$taskDashboard = "KHQuant Dashboard"
$taskUpdate = "KHQuant Daily Update"

function Install-Services {
    $batPath = Join-Path $RepoRoot "khquant-native.bat"
    if (-not (Test-Path -LiteralPath $batPath -PathType Leaf)) {
        throw "KHQuant launcher not found under repository root: $batPath"
    }
    New-Item -ItemType Directory -Path (Join-Path $RepoRoot ".khquant\logs") -Force | Out-Null

    Uninstall-Services -Silent

    $actionDashboard = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument ('/d /c ""{0}" dashboard"' -f $batPath) `
        -WorkingDirectory $RepoRoot
    $actionUpdate = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument ('/d /c ""{0}" update-data --mode local"' -f $batPath) `
        -WorkingDirectory $RepoRoot
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

    $triggerDashboard = New-ScheduledTaskTrigger -AtLogon -User "$env:USERNAME"
    Register-ScheduledTask -TaskName $taskDashboard -Action $actionDashboard -Trigger $triggerDashboard -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "Registered: $taskDashboard (logon trigger)"

    $triggerUpdate = New-ScheduledTaskTrigger -Daily -At 09:00
    Register-ScheduledTask -TaskName $taskUpdate -Action $actionUpdate -Trigger $triggerUpdate -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "Registered: $taskUpdate (daily 09:00)"
}

function Uninstall-Services {
    param([switch]$Silent)
    foreach ($name in @($taskDashboard, $taskUpdate)) {
        $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            if (-not $Silent) { Write-Host "Removed: $name" }
        }
    }
}

if ($Uninstall) {
    Uninstall-Services
} else {
    Install-Services
}
