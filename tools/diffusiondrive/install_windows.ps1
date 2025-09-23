param(
  [string]$InstallDir = "$PSScriptRoot\pc",
  [string]$Python = "python",
  [string]$TorchPackages = "torch torchvision torchaudio",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu118",
  [switch]$SkipWeights,
  [string]$HuggingFaceToken
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    throw "Path must not be empty."
  }

  $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
  if ($null -ne $resolved) {
    return [System.IO.Path]::GetFullPath($resolved.Path)
  }

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  $combined = Join-Path -Path (Get-Location) -ChildPath $Path
  return [System.IO.Path]::GetFullPath($combined)
}

function Run-Step {
  param(
    [string]$Message,
    [ScriptBlock]$Action
  )
  Write-Host "[+] $Message"
  & $Action
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

$pythonCmd = (Get-Command $Python -ErrorAction Stop).Source
$gitCmd = (Get-Command git -ErrorAction Stop).Source

$pythonVersion = & $pythonCmd -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ([version]$pythonVersion -lt [version]"3.10") {
  throw "Python 3.10 or newer is required (found $pythonVersion)."
}

$installDir = Resolve-AbsolutePath -Path $InstallDir
if (-not (Test-Path $installDir)) {
  New-Item -ItemType Directory -Path $installDir | Out-Null
}

$diffRepo = Join-Path $installDir "DiffusionDrive"
$venvDir = Join-Path $installDir "dd-env"
$weightsDir = Join-Path $installDir "weights"

Run-Step -Message "Cloning/updating DiffusionDrive repo" -Action {
  if (-not (Test-Path $diffRepo)) {
    & $gitCmd clone --depth 1 https://github.com/hustvl/DiffusionDrive.git $diffRepo
  } else {
    Push-Location $diffRepo
    try {
      & $gitCmd fetch --tags
      & $gitCmd pull --ff-only
    } finally {
      Pop-Location
    }
  }
  Push-Location $diffRepo
  try {
    & $gitCmd lfs install | Out-Null
  } catch {
    Write-Warning "git-lfs not available; continuing without local LFS filters."
  } finally {
    Pop-Location
  }
}

Run-Step -Message "Creating virtual environment" -Action {
  if (-not (Test-Path $venvDir)) {
    & $pythonCmd -m venv $venvDir
  }
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment python not found at $venvPython"
}

Run-Step -Message "Upgrading pip" -Action {
  & $venvPython -m pip install --upgrade pip
}

$torchPackagesList = @()
if (-not [string]::IsNullOrWhiteSpace($TorchPackages)) {
  $torchPackagesList = $TorchPackages -split '\s+'
}
if ($torchPackagesList.Count -gt 0) {
  Run-Step -Message "Installing PyTorch stack" -Action {
    $args = @("install", "--upgrade") + $torchPackagesList
    if (-not [string]::IsNullOrWhiteSpace($TorchIndexUrl)) {
      $args += @("--index-url", $TorchIndexUrl)
    }
    & $venvPython -m pip @args
  }
}

Run-Step -Message "Installing DiffusionDrive requirements" -Action {
  $requirements = Join-Path $diffRepo "requirements.txt"
  if (Test-Path $requirements) {
    & $venvPython -m pip install -r $requirements
  } else {
    Write-Warning "requirements.txt not found at $requirements"
  }
  & $venvPython -m pip install -r (Join-Path $repoRoot "tools\diffusiondrive\server\requirements.txt")
}

if (-not $SkipWeights) {
  Run-Step -Message "Downloading Hugging Face weights" -Action {
    if ($HuggingFaceToken) {
      $env:HUGGINGFACEHUB_API_TOKEN = $HuggingFaceToken
    }
    & $venvPython (Join-Path $repoRoot "tools\diffusiondrive\server\download_weights.py") --local-dir $weightsDir
    if ($HuggingFaceToken) {
      Remove-Item Env:HUGGINGFACEHUB_API_TOKEN -ErrorAction SilentlyContinue
    }
  }
}

Write-Host ""
Write-Host "Setup complete. Key locations:"
Write-Host "  Install directory : $installDir"
Write-Host "  Repo checkout     : $diffRepo"
Write-Host "  Virtual env       : $venvDir"
if (-not $SkipWeights) {
  Write-Host "  Weights           : $weightsDir"
}
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "  & `"$venvDir\Scripts\Activate.ps1`""
Write-Host "Start the sample server:"
Write-Host "  & `"$venvPython`" `"$repoRoot\tools\diffusiondrive\server\diffusiondrive_server.py`" --bind tcp://0.0.0.0:5555"

Write-Host ""
Write-Host "If the bridge runs on a different host, ensure TCP port 5555 is allowed through the Windows firewall."
