$ErrorActionPreference = 'Stop'

Write-Output "Opening Windows Update settings…"
Start-Process "ms-settings:windowsupdate" | Out-Null
