$ErrorActionPreference = 'Stop'

$agentsDir = Join-Path $PSScriptRoot '..\.github\agents'
$agentFiles = Get-ChildItem -LiteralPath $agentsDir -Filter '*.agent.md' -File -ErrorAction Stop

if (-not $agentFiles -or $agentFiles.Count -eq 0) {
  Write-Output "No workspace agents found under $agentsDir"
  exit 0
}

$hadErrors = $false

foreach ($file in $agentFiles) {
  Write-Output ""
  Write-Output "== $($file.Name) =="

  $bytes = [IO.File]::ReadAllBytes($file.FullName)
  $hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)

  # Detect CRLF by scanning raw bytes
  $hasCrlf = $false
  for ($i = 0; $i -lt $bytes.Length - 1; $i++) {
    if ($bytes[$i] -eq 0x0D -and $bytes[$i + 1] -eq 0x0A) { $hasCrlf = $true; break }
  }

  $text = Get-Content -LiteralPath $file.FullName -Raw

  # Support both plain frontmatter and the repo's ```chatagent fenced format.
  $normalized = $text
  if ($normalized -match '(?s)^```chatagent\n(.*)\n```\s*\z') {
    $normalized = $Matches[1]
  }

  $frontmatterMatch = $normalized -match "(?s)^---\n(.*?)\n---\n(.*)\z"

  Write-Output "bom=$hasBom crlf=$hasCrlf frontmatterMatch=$frontmatterMatch"

  if ($hasBom) {
    Write-Output 'ERROR: UTF-8 BOM detected (may break parsing)'
    $hadErrors = $true
  }

  if ($hasCrlf) {
    Write-Output 'ERROR: CRLF line endings detected (may break parsing)'
    $hadErrors = $true
  }

  if (-not $frontmatterMatch) {
    $head = $normalized.Substring(0, [Math]::Min(220, $normalized.Length)) -replace "\n", "\\n"
    Write-Output "ERROR: Missing/invalid YAML frontmatter. head=$head"
    $hadErrors = $true
    continue
  }

  $fm = $Matches[1]
  $required = @('name:', 'description:', 'tools:', 'target:')
  foreach ($req in $required) {
    if (-not ($fm -split "\n" | Where-Object { $_.TrimStart().StartsWith($req) })) {
      Write-Output "ERROR: Frontmatter missing $req"
      $hadErrors = $true
    }
  }
}

if ($hadErrors) {
  exit 1
}

Write-Output "\nOK: workspace agents look well-formed"
