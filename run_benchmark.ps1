$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$benchDir = Join-Path $repoRoot "benchmark"

if (-not (Test-Path (Join-Path $benchDir "run_benchmark.ps1"))) {
  throw "Missing benchmark script: $benchDir\\run_benchmark.ps1"
}

& (Join-Path $benchDir "run_benchmark.ps1") @args

