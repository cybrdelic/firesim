$ErrorActionPreference = 'Stop'

Write-Output "== FireSim WebGPU Fix Tool =="
Write-Output "This runs the common diagnostics and opens the right system pages."
Write-Output ""

Write-Output "(1/4) Checking dxil.dll…"
& "$PSScriptRoot\check-dxil.ps1"

Write-Output ""
Write-Output "(2/4) Checking Chrome/Edge channels…"
& "$PSScriptRoot\check-chrome.ps1"

Write-Output ""
Write-Output "(3/4) Opening Windows Update…"
& "$PSScriptRoot\open-windows-update.ps1"

Write-Output ""
Write-Output "(4/4) Opening GPU driver download…"
& "$PSScriptRoot\open-gpu-driver.ps1"

Write-Output ""
Write-Output "Next:"
Write-Output "- After updating Windows/driver, rerun: pnpm -s gpu:health:headed"
Write-Output "- Or run against Chrome Stable: pnpm -s gpu:health:chrome"
