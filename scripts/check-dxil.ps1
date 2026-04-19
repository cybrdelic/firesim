$ErrorActionPreference = 'Stop'

function Get-FileVersionSafe([string]$path) {
  try {
    return (Get-Item -LiteralPath $path -ErrorAction Stop).VersionInfo.FileVersion
  } catch {
    return $null
  }
}

Write-Output "== DXIL/DXC check =="

$dxilPaths = @(
  "$env:WINDIR\System32\dxil.dll",
  "$env:WINDIR\SysWOW64\dxil.dll"
)

foreach ($p in $dxilPaths) {
  $exists = Test-Path -LiteralPath $p
  if ($exists) {
    $ver = Get-FileVersionSafe $p
    Write-Output "FOUND: $p"
    if ($ver) { Write-Output "  version=$ver" }
  } else {
    Write-Output "MISSING: $p"
  }
}

Write-Output ""
Write-Output "Notes:"
Write-Output "- If dxil.dll is missing/corrupt, WebGPU device creation can fail in Chromium (Dawn/D3D12)."
Write-Output "- Next tools:"
Write-Output "  - Open Windows Update: pnpm -s windows:update"
Write-Output "  - Open GPU driver download: pnpm -s gpu:driver"
