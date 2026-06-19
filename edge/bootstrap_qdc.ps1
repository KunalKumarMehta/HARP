<#
  HARP · edge/bootstrap_qdc.ps1 · MIT
  One-shot provisioner for a BAREBONE Qualcomm Device Cloud Snapdragon X Elite
  (Windows on ARM64). A fresh QDC interactive session ships with no Python, no
  pip deps, and genie-t2t-run not on PATH — this turns it into a box that can run
  the Risk-A gate, idempotently, in one command.

  Run it (from the repo root, in the SSH/cmd or PowerShell session):
      edge\bootstrap_qdc.cmd                 # cmd.exe one-liner (recommended)
  or directly:
      powershell -ExecutionPolicy Bypass -File edge\bootstrap_qdc.ps1

  What it does, detect-first and idempotent (safe to re-run):
    1. Python 3.12 ARM64 — detect, else download+silently install from python.org.
    2. .venv + HARP deps (websockets, jsonschema, numpy).
    3. genie-t2t-run — search the machine for the QAIRT Genie runtime that QDC
       images stage; wire it onto PATH and pin HARP_GENIE_BIN. If absent, accept a
       QAIRT zip via -QairtZip / $env:HARP_QAIRT_ZIP, else print exact next steps.
    4. Verify everything and (unless -SkipRun) run the Risk-A gate (run_test.py).

  Flags:
    -SkipRun      provision only, don't run the gate
    -WithOnnx     also pip-install onnxruntime-genai/-qnn (only for the SELF-compile
                  QNNBackend path; NOT needed for the precompiled Genie bundle)
    -QairtZip <p> path or URL to a QAIRT SDK zip if the box has none staged
#>
[CmdletBinding()]
param(
  [string]$PythonVersion = "3.12.7",
  [string]$QairtZip = $env:HARP_QAIRT_ZIP,
  [string]$QpmExe = "",          # path to QPM installer exe; auto-detected if blank
  [string]$QairtPackage = "qualcomm-ai-runtime",
  [string]$QairtVersion = "2.45",
  [switch]$WithOnnx,
  [switch]$SkipRun
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root = Split-Path $PSScriptRoot -Parent           # repo root (this script is in edge/)
function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!!  $m" -ForegroundColor Yellow }
function Ok($m)  { Write-Host "ok  $m" -ForegroundColor Green }

function Refresh-Path {
  $m = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $u = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = (@($m, $u) | Where-Object { $_ }) -join ";"
}

Say "HARP QDC bootstrap — repo root: $Root"

# ---- 1. Python -------------------------------------------------------------
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  $boot = (Get-Command python -ErrorAction SilentlyContinue).Source
  if (-not $boot) {
    Say "Python not found — installing $PythonVersion (ARM64) from python.org"
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-arm64.exe"
    $exe = Join-Path $env:TEMP "python-$PythonVersion-arm64.exe"
    Invoke-WebRequest -Uri $url -OutFile $exe
    Start-Process -FilePath $exe -Wait -ArgumentList `
      "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1", "Include_launcher=1"
    Refresh-Path
    $boot = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $boot) { throw "Python install completed but 'python' still not on PATH. Open a new shell and re-run." }
  }
  Ok "bootstrap python: $boot"
  Say "Creating .venv"
  & $boot -m venv (Join-Path $Root ".venv")
}
Ok "venv python: $venvPy"

# ---- 2. deps ---------------------------------------------------------------
Say "Installing HARP runtime deps"
& $venvPy -m pip install --disable-pip-version-check --upgrade pip | Out-Null
& $venvPy -m pip install --disable-pip-version-check websockets jsonschema numpy
if ($WithOnnx) {
  Say "Installing onnxruntime-genai / -qnn (self-compile path)"
  & $venvPy -m pip install --disable-pip-version-check onnxruntime-genai onnxruntime-qnn
}
Ok "deps installed"

# ---- 2b. QPM / QPM-CLI (install QAIRT when not pre-staged) -----------------
function Find-QpmCli {
  $qpmCli = Get-Command qpm-cli -ErrorAction SilentlyContinue
  if ($qpmCli) { return $qpmCli.Source }
  # QPM installs into AppData\Local\QPM by default
  $candidate = Join-Path $env:LOCALAPPDATA "QPM\qpm-cli.exe"
  if (Test-Path $candidate) { return $candidate }
  return $null
}

function Install-Qpm {
  # Locate the QPM installer: -QpmExe flag > repo root > Downloads
  $candidates = @(
    $QpmExe,
    (Join-Path $Root "QPM*.exe"),
    (Join-Path $env:USERPROFILE "Downloads\QPM*.exe")
  ) | Where-Object { $_ }

  $exe = $null
  foreach ($pat in $candidates) {
    $hit = Get-Item $pat -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $exe = $hit.FullName ; break }
  }

  if (-not $exe) {
    Warn "QPM installer not found. Drop QPM*.exe next to the repo root, or pass -QpmExe <path>."
    Warn "Download from: https://qpm.qualcomm.com  (free account required)"
    return $false
  }

  Say "Installing QPM from: $exe"
  Start-Process -FilePath $exe -Wait -ArgumentList "/quiet", "/norestart"
  Refresh-Path
  return $true
}

function Install-QairtViaQpm($cli) {
  Say "Using qpm-cli to install $QairtPackage $QairtVersion"
  # qpm-cli uses stored credentials (set once with: qpm-cli --notty login)
  # If not logged in it will prompt; that's OK for an interactive bootstrap.
  $result = & $cli package install $QairtPackage --version $QairtVersion 2>&1
  if ($LASTEXITCODE -ne 0) {
    Warn "qpm-cli install failed (exit $LASTEXITCODE). Output:"
    $result | ForEach-Object { Warn "  $_" }
    Warn "If credentials are needed, run once interactively:"
    Warn "  qpm-cli --notty login"
    Warn "Then re-run this script."
    return $false
  }
  Ok "QAIRT installed via qpm-cli"
  return $true
}

# Only try QPM when genie isn't already present and no zip was given
$qpmCli = Find-QpmCli
if (-not $qpmCli) {
  Say "qpm-cli not found — attempting QPM installation"
  if (Install-Qpm) {
    Refresh-Path
    $qpmCli = Find-QpmCli
    if ($qpmCli) { Ok "qpm-cli: $qpmCli" }
    else { Warn "QPM installed but qpm-cli still not on PATH. Open a new shell and re-run." }
  }
} else {
  Ok "qpm-cli: $qpmCli"
}

# ---- 3. genie-t2t-run (QAIRT Genie runtime) --------------------------------
function Find-Genie {
  $roots = @(
    $env:QNN_SDK_ROOT, $env:QAIRT_SDK_ROOT, $env:QAIRT_ROOT,
    "C:\Qualcomm", "C:\Program Files\Qualcomm", "C:\ProgramData\Qualcomm",
    "C:\opt\qcom", $env:LOCALAPPDATA, $env:USERPROFILE,
    (Join-Path $env:USERPROFILE "Downloads"), (Join-Path $Root ".qairt")
  ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
  foreach ($r in $roots) {
    $hit = Get-ChildItem -Path $r -Recurse -Filter "genie-t2t-run.exe" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  return $null
}

$genie = Find-Genie

# Try QPM-CLI first (before zip fallback) when QAIRT isn't staged
if (-not $genie -and -not $QairtZip -and $qpmCli) {
  if (Install-QairtViaQpm $qpmCli) { $genie = Find-Genie }
}

if (-not $genie -and $QairtZip) {
  Say "genie-t2t-run not staged — provisioning QAIRT from: $QairtZip"
  $dest = Join-Path $Root ".qairt"
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  if ($QairtZip -match '^https?://') {
    $zip = Join-Path $env:TEMP "qairt.zip"
    Invoke-WebRequest -Uri $QairtZip -OutFile $zip
  } else { $zip = $QairtZip }
  Expand-Archive -Path $zip -DestinationPath $dest -Force
  $genie = Find-Genie
}

if ($genie) {
  $dir = Split-Path $genie -Parent
  $env:Path = "$dir;$env:Path"
  setx HARP_GENIE_BIN "$genie" | Out-Null      # persist for future sessions
  $env:HARP_GENIE_BIN = $genie                 # and this one
  Ok "genie-t2t-run: $genie"
} else {
  Warn "genie-t2t-run not found. It ships in QAIRT SDK $QairtVersion."
  Warn "Options (tried in order by this script):"
  Warn "  1. QPM-CLI auto-install: drop QPM*.exe in repo root OR pass -QpmExe <path>."
  Warn "     If QPM needs login first:  qpm-cli --notty login  then re-run."
  Warn "  2. Zip path/URL:  edge\bootstrap_qdc.cmd -QairtZip C:\path\to\qairt-$QairtVersion.zip"
  Warn "  3. On QDC the AI stack may still be provisioning — re-run in a few minutes."
  Warn "Download QPM (free account): https://qpm.qualcomm.com"
  if (-not $SkipRun) { Warn "Skipping the gate run (no runtime). Re-run once QAIRT is present." ; $SkipRun = $true }
}

# ---- 4. verify + run -------------------------------------------------------
Say "Environment summary"
& $venvPy --version
& $venvPy -m pip show websockets 2>$null | Select-String "Version" | ForEach-Object { Write-Host "    websockets $_" }
$bundle = Join-Path $Root "build\qwen3-4b-w4a16\genie_config.json"
if (Test-Path $bundle) { Ok "precompiled bundle present: build\qwen3-4b-w4a16" }
else { Warn "precompiled bundle missing at build\qwen3-4b-w4a16 — scp/clone it onto the box." }
if ($env:HARP_GENIE_BIN) { Ok "HARP_GENIE_BIN=$env:HARP_GENIE_BIN" }

if (-not $SkipRun) {
  Say "Running Risk-A gate: python run_test.py"
  Push-Location $Root
  try { & $venvPy run_test.py } finally { Pop-Location }
} else {
  Say "Provisioning done. Run the gate yourself with:  .venv\Scripts\python run_test.py"
}
