param(
  [int]$Port = 4173,
  [string]$HostAddress = '127.0.0.1',
  [switch]$StartServer
)

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

function Stop-ProcessOnPort([int]$p) {
  try {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop
  } catch {
    return
  }

  foreach ($c in $conns) {
    if (-not $c.OwningProcess) { continue }
    try {
      $proc = Get-Process -Id $c.OwningProcess -ErrorAction Stop
      Write-Output "Stopping process $($proc.ProcessName) (pid=$($proc.Id)) on port $p"
      Stop-Process -Id $proc.Id -Force -ErrorAction Stop
    } catch {
      # ignore
    }
  }
}

Write-Output "== FireSim full reload =="
Write-Output "repoRoot=$repoRoot"
Write-Output "host=$HostAddress port=$Port"

Stop-ProcessOnPort -p $Port

$pathsToClear = @(
  (Join-Path $repoRoot 'node_modules/.vite'),
  (Join-Path $repoRoot '.vite'),
  (Join-Path $repoRoot 'dist'),
  (Join-Path $repoRoot 'test-results/gpu-health.json')
)

foreach ($p in $pathsToClear) {
  if (Test-Path -LiteralPath $p) {
    Write-Output "Removing $p"
    Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Write-Output "Done clearing caches."

if ($StartServer) {
  Write-Output "Starting dev server (forced re-optimize)…"
  $args = @('dev', '--', '--host', $HostAddress, '--port', "$Port", '--force')
  Start-Process -FilePath 'pnpm' -ArgumentList $args -WorkingDirectory $repoRoot | Out-Null
  Write-Output "Dev server launched."
}
