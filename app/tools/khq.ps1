$ErrorActionPreference = "Stop"
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
& (Join-Path $taskRoot "khquant-native.bat") @args
exit $LASTEXITCODE
