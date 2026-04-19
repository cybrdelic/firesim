$ErrorActionPreference = 'Stop'

$paths = @(
  'C:/Users/alexf/.copilot/agents/agent-architect.agent.md',
  'C:/Users/alexf/.copilot/agents/dead-code-analyst.agent.md',
  'C:/Users/alexf/.copilot/agents/career-agent.agent.md'
)

foreach ($p in $paths) {
  Write-Output ""
  Write-Output "== $([IO.Path]::GetFileName($p)) =="

  if (-not (Test-Path $p)) {
    Write-Output "missing"
    continue
  }

  $bytes = [IO.File]::ReadAllBytes($p)
  $hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)

  # Detect CRLF by scanning raw bytes
  $hasCrlf = $false
  for ($i = 0; $i -lt $bytes.Length - 1; $i++) {
    if ($bytes[$i] -eq 0x0D -and $bytes[$i + 1] -eq 0x0A) { $hasCrlf = $true; break }
  }

  $text = Get-Content -LiteralPath $p -Raw
  $frontmatterMatch = $text -match "(?s)^---\n(.*?)\n---\n(.*)\z"

  Write-Output "bom=$hasBom crlf=$hasCrlf frontmatterMatch=$frontmatterMatch"

  if ($frontmatterMatch) {
    $fm = $Matches[1]
    $toolsLines = ($fm -split "\n" | Where-Object { $_ -match '^tools\s*:' })
    Write-Output ("toolsLines=" + ($toolsLines -join ' | '))
  }
  else {
    $head = $text.Substring(0, [Math]::Min(200, $text.Length)) -replace "\n", "\\n"
    Write-Output "head=$head"
  }
}
