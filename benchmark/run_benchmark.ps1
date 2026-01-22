$ErrorActionPreference = "Stop"

$benchmarkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $benchmarkDir
Set-Location $benchmarkDir

$venvDir = Join-Path $benchmarkDir ".venv"
$py = Join-Path $venvDir "Scripts\\python.exe"

if (-not (Test-Path $py)) {
  Write-Host "Creating venv: $venvDir"
  python -m venv $venvDir
}

Write-Host "Installing FlashToken (editable) + bench deps..."
& $py -m pip -q install --upgrade pip
& $py -m pip -q install -r (Join-Path $benchmarkDir "requirements.txt")

Write-Host "Running benchmark suite..."
& $py (Join-Path $benchmarkDir "run.py") @args

