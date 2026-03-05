$ErrorActionPreference = 'Stop'

function Try-GetVersion([string]$exePath) {
  try {
    if (-not (Test-Path -LiteralPath $exePath)) { return $null }
    $p = Start-Process -FilePath $exePath -ArgumentList @('--version') -NoNewWindow -PassThru -RedirectStandardOutput -Wait -ErrorAction Stop
    return $p.StandardOutput.ReadToEnd().Trim()
  } catch {
    return $null
  }
}

Write-Output "== Chrome channel check =="

$candidates = @(
  @{ name = 'chrome';  path = "$env:ProgramFiles\Google\Chrome\Application\chrome.exe" },
  @{ name = 'chrome-x86'; path = "$env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe" },
  @{ name = 'chrome-beta'; path = "$env:ProgramFiles\Google\Chrome Beta\Application\chrome.exe" },
  @{ name = 'chrome-dev';  path = "$env:ProgramFiles\Google\Chrome Dev\Application\chrome.exe" },
  @{ name = 'chrome-canary'; path = "$env:LOCALAPPDATA\Google\Chrome SxS\Application\chrome.exe" },
  @{ name = 'msedge'; path = "$env:ProgramFiles(x86)\Microsoft\Edge\Application\msedge.exe" },
  @{ name = 'msedge'; path = "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe" }
)

$found = $false
foreach ($c in $candidates) {
  $ver = Try-GetVersion $c.path
  if ($ver) {
    $found = $true
    Write-Output "FOUND: $($c.name)"
    Write-Output "  path=$($c.path)"
    Write-Output "  version=$ver"
  }
}

if (-not $found) {
  Write-Output "No Chrome/Edge installs detected via common paths."
}

Write-Output ""
Write-Output "Tip: you can run the health tool against Chrome Stable via:"
Write-Output "  pnpm -s gpu:health:chrome"
