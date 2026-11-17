param([int]$Port = 8015)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path "data/boundary_v15/manifest.json")) {
    throw "Missing result bundle. Download curio-boundary-v15-windows.zip and extract it first."
}
$Python = Join-Path (Get-Location) ".venv-boundary/Scripts/python.exe"
if (!(Test-Path $Python)) {
    py -3.12 -m venv .venv-boundary
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 for Windows, then retry." }
}
& $Python -m pip install -r requirements-boundary-demo.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
$env:PYTHONPATH = Join-Path (Get-Location) "src"
& $Python -m curio.serve.boundary_snapshot --port $Port
exit $LASTEXITCODE
