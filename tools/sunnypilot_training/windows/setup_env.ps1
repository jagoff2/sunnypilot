Param(
  [string]$PythonVersion = "3.11.6",
  [switch]$InstallCUDA,
  [string]$CarlaVersion = "0.10.0",
  [string]$InstallDir = "$PSScriptRoot/../../.."
)

$ErrorActionPreference = "Stop"

function Install-Python {
  param(
    [string]$Version,
    [string]$Destination
  )
  Write-Host "Installing Python $Version"
  $pythonInstaller = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
  $installerPath = Join-Path $env:TEMP "python-$Version.exe"
  Invoke-WebRequest -Uri $pythonInstaller -OutFile $installerPath
  Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 TargetDir=$Destination" -Wait
}

function Ensure-Venv {
  param(
    [string]$PythonExe
  )
  $venvPath = Join-Path $PSScriptRoot "..\..\..\venv"
  if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $PythonExe -m venv $venvPath
  }
  return $venvPath
}

function Install-PythonPackages {
  param(
    [string]$VenvPath
  )
  $pip = Join-Path $VenvPath "Scripts\pip.exe"
  & $pip install --upgrade pip
  & $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  & $pip install hydra-core omegaconf tinygrad==0.8.0 zarr numpy scipy tqdm
  & $pip install carla==$CarlaVersion pygame
}

function Install-CARLA {
  param(
    [string]$Version,
    [string]$Destination
  )
  $carlaZip = "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_$Version.tar.gz"
  $downloadPath = Join-Path $env:TEMP "CARLA_$Version.tar.gz"
  Invoke-WebRequest -Uri $carlaZip -OutFile $downloadPath
  Write-Host "Extracting CARLA to $Destination"
  if (-Not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
  }
  tar -xf $downloadPath -C $Destination
}

$pythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Path
if (-Not $pythonPath) {
  $targetDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$($PythonVersion.Replace('.', ''))"
  Install-Python -Version $PythonVersion -Destination $targetDir
  $pythonPath = Join-Path $targetDir "python.exe"
}

$venv = Ensure-Venv -PythonExe $pythonPath
Install-PythonPackages -VenvPath $venv

if (-Not (Test-Path "$InstallDir\CARLA_$CarlaVersion")) {
  Install-CARLA -Version $CarlaVersion -Destination $InstallDir
}

Write-Host "Environment setup complete. Activate with:`n`""$venv\Scripts\Activate.ps1""`
