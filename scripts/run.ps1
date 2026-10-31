[CmdletBinding()]
param(
    [ValidateSet('siglip', 'clip')]
    [string]$Encoder = 'siglip',
    [string]$Index = '',
    [string]$Data = 'data/snapshots/collection_v2.jsonl',
    [string]$Probes = '',
    [ValidateRange(-1, 1)]
    [double]$ProbeWeight = -1,
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Missing .venv. Run uv sync --extra dev from the repository root.'
}
if (-not (Test-Path -LiteralPath $Data -PathType Leaf)) {
    throw "Missing catalog $Data. Run .\.venv\Scripts\python.exe scripts/data_ingest.py and freeze the completed catalog first."
}
$dataHash = (Get-FileHash -LiteralPath $Data -Algorithm SHA256).Hash.ToLowerInvariant()
$useVisualSelection = $false
$visualSelection = $null
$visualChoice = $null
if (-not $PSBoundParameters.ContainsKey('Encoder') -and -not $PSBoundParameters.ContainsKey('Index') -and
    (Test-Path -LiteralPath 'artifacts/visual_selection.json')) {
    $visualSelection = Get-Content -LiteralPath 'artifacts/visual_selection.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($visualSelection.data_sha256 -eq $dataHash) {
        $visualChoice = $visualSelection.selected
        if ($visualChoice.variant -ne 'image_only') { throw 'Frozen production choice must use image_only.' }
        if ($visualChoice.model -notin @('clip', 'siglip', 'siglip_adapted', 'siglip_visual_adapted')) {
            throw "Frozen production choice has an invalid model: $($visualChoice.model)"
        }
        $Encoder = if ($visualChoice.model -eq 'clip') { 'clip' } else { 'siglip' }
        $Index = "artifacts/$($visualChoice.model)"
        if (-not $PSBoundParameters.ContainsKey('Probes')) { $Probes = [string]$visualChoice.probes }
        if ($ProbeWeight -lt 0) { $ProbeWeight = [double]$visualChoice.weight }
        $useVisualSelection = $true
    }
}
if ([string]::IsNullOrWhiteSpace($Index)) { $Index = "artifacts/$Encoder" }
foreach ($name in @('image_embeddings.npy', 'item_ids.json', 'manifest.json')) {
    $indexFile = Join-Path $Index $name
    if (-not (Test-Path -LiteralPath $indexFile -PathType Leaf)) {
        throw "Missing index $indexFile. Run .\.venv\Scripts\python.exe -m curio.index.build --encoder $Encoder --data `"$Data`" --output `"$Index`"."
    }
}
$manifest = Get-Content -LiteralPath (Join-Path $Index 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.encoder -ne $Encoder) {
    throw "Index encoder is $($manifest.encoder), but requested $Encoder. Use matching -Encoder and -Index values."
}
if ($manifest.data_sha256 -ne $dataHash) {
    throw 'Catalog/index hash mismatch. Use the frozen catalog matching this index or rebuild the index.'
}
if ($useVisualSelection) {
    if ($null -eq $visualSelection.model_files_sha256) {
        throw 'Frozen visual selection lacks model_files_sha256; refuse to start the default production configuration.'
    }
    function Assert-FrozenVisualArtifact([string]$Artifact) {
        $relative = $Artifact.Replace('\', '/')
        $entry = $visualSelection.model_files_sha256.PSObject.Properties[$relative]
        if ($null -eq $entry) { throw "Frozen visual artifact lacks a fingerprint: $relative" }
        $path = if ([IO.Path]::IsPathRooted($Artifact)) { $Artifact } else { Join-Path $repoRoot $Artifact }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Frozen visual artifact is missing: $relative"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$entry.Value).ToLowerInvariant()) {
            throw "Frozen visual artifact changed: $relative"
        }
    }
    foreach ($name in @('image_embeddings.npy', 'item_ids.json', 'manifest.json')) {
        Assert-FrozenVisualArtifact (Join-Path $Index $name)
    }
    if ($manifest.adapted) {
        if ([string]::IsNullOrWhiteSpace([string]$manifest.adapter_path)) {
            throw 'The selected adapted index has no adapter_path.'
        }
        foreach ($name in @('adapter_config.json', 'adapter_model.safetensors')) {
            Assert-FrozenVisualArtifact (Join-Path $manifest.adapter_path $name)
        }
    }
}
if (-not $PSBoundParameters.ContainsKey('Probes') -and [string]::IsNullOrWhiteSpace($Probes) -and $Index -eq 'artifacts/siglip' -and (Test-Path -LiteralPath 'artifacts/probes/siglip/report.json')) {
    $Probes = 'artifacts/probes/siglip'
}
if (-not [string]::IsNullOrWhiteSpace($Probes)) {
    $probeReport = Get-Content -LiteralPath (Join-Path $Probes 'report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($probeReport.data_sha256 -ne $dataHash -or $probeReport.index_sha256 -ne $manifest.index_sha256) {
        throw 'Probe/data/index mismatch. Train probes on this exact catalog and index.'
    }
    if ($useVisualSelection -and -not $PSBoundParameters.ContainsKey('Probes')) {
        foreach ($name in @('report.json', 'ontology.json')) {
            Assert-FrozenVisualArtifact (Join-Path $Probes $name)
        }
        foreach ($task in $probeReport.tasks.PSObject.Properties) {
            if ($task.Value.status -eq 'trained') {
                Assert-FrozenVisualArtifact (Join-Path $Probes "$($task.Name).npz")
            }
        }
    }
}
if ($ProbeWeight -lt 0) {
    $ProbeWeight = 0.15
    if (Test-Path -LiteralPath 'artifacts/selection.json') {
        $selection = Get-Content -LiteralPath 'artifacts/selection.json' -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($selection.data_sha256 -eq $dataHash) { $ProbeWeight = [double]$selection.probe_weight }
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'web/dist/index.html') -PathType Leaf)) {
    throw 'Missing frontend build. Run pnpm --dir web install --frozen-lockfile and pnpm --dir web build.'
}

$env:CURIO_ENCODER = $Encoder
$env:CURIO_INDEX = $Index
$env:CURIO_DATA = $Data
$env:CURIO_PROBES = $Probes
$env:CURIO_PROBE_WEIGHT = [string]$ProbeWeight
$env:CURIO_HOST = '127.0.0.1'
$env:CURIO_PORT = [string]$Port
$usingFrozenProduction = $useVisualSelection -and -not $PSBoundParameters.ContainsKey('Probes') -and
    $ProbeWeight -eq [double]$visualChoice.weight
if ($usingFrozenProduction) {
    $env:CURIO_METRICS = 'artifacts/evaluation/image_audit/selected_test/metrics.json'
} else {
    Remove-Item Env:CURIO_METRICS -ErrorAction SilentlyContinue
}
# This launcher never initiates model downloads or training, including requests
# that arrive after startup. Explicit index-build commands prepare the cache.
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

Write-Host "Curio: encoder=$Encoder index=$Index data=$Data probes=$Probes probeWeight=$ProbeWeight"
if ($usingFrozenProduction) { Write-Host 'Using frozen visual-dev production selection. Default retrieval: image_only.' }
elseif ($useVisualSelection) { Write-Host 'Using the frozen visual-dev image index with explicit probe settings.' }
Write-Host 'Offline mode. If the matching model is not cached, searches explicitly report degraded status.'
if ($CheckOnly) {
    Write-Host 'Resource checks passed.'
    exit 0
}
Write-Host "Open http://127.0.0.1:$Port . Press Ctrl+C to stop."
& $pythonPath -m curio.serve
exit $LASTEXITCODE
