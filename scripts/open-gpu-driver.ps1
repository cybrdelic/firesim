$ErrorActionPreference = 'Stop'

Write-Output "== GPU driver helper =="

$controllers = @()
try {
  $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
} catch {
  Write-Output "Could not query GPU via WMI: $($_.Exception.Message)"
}

$name = ($controllers | Select-Object -First 1 -ExpandProperty Name -ErrorAction SilentlyContinue)
if (-not $name) { $name = '' }

Write-Output "Detected GPU: $name"

$lower = $name.ToLowerInvariant()

if ($lower -match 'nvidia') {
  Write-Output "Opening NVIDIA driver download…"
  Start-Process "https://www.nvidia.com/Download/index.aspx" | Out-Null
  exit 0
}

if ($lower -match 'amd|radeon') {
  Write-Output "Opening AMD driver download…"
  Start-Process "https://www.amd.com/en/support" | Out-Null
  exit 0
}

if ($lower -match 'intel') {
  Write-Output "Opening Intel driver download…"
  Start-Process "https://www.intel.com/content/www/us/en/download-center/home.html" | Out-Null
  exit 0
}

Write-Output "Opening generic driver help…"
Start-Process "https://support.microsoft.com/windows/update-drivers-manually-in-windows" | Out-Null
