Param(
  [string]$PythonVersion = "3.7.9",
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
  Remove-Item $installerPath -ErrorAction SilentlyContinue
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
    [string]$VenvPath,
    [switch]$UseCUDA
  )

  $pip = Join-Path $VenvPath "Scripts\pip.exe"
  & $pip install --upgrade pip

  if ($UseCUDA) {
    Write-Host "Installing CUDA-enabled PyTorch 1.12.1 (cu113)"
    $torchIndex = "https://download.pytorch.org/whl/cu113"
    & $pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url $torchIndex
  }
  else {
    Write-Host "Installing CPU PyTorch 1.12.1"
    & $pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
  }

  & $pip install hydra-core==1.3.2 omegaconf==2.3.0 tinygrad==0.8.0 zarr numpy scipy tqdm pygame pytest
}

function Install-CARLA {
  param(
    [string]$Version,
    [string]$Destination
  )

  $carlaZip = "https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/CARLA_$Version.zip"
  $downloadPath = Join-Path $env:TEMP "CARLA_$Version.zip"
  Write-Host "Downloading CARLA $Version"
  Invoke-WebRequest -Uri $carlaZip -OutFile $downloadPath
  Write-Host "Extracting CARLA archive to $Destination"
  if (-Not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
  }
  Expand-Archive -Path $downloadPath -DestinationPath $Destination -Force
  Remove-Item $downloadPath -ErrorAction SilentlyContinue
}

function Install-CarlaPythonModule {
  param(
    [string]$VenvPath,
    [string]$InstallRoot,
    [string]$Version
  )

  $pythonApi = Join-Path $InstallRoot "CARLA_$Version\PythonAPI"
  $distDir = Join-Path $pythonApi "carla\dist"
  if (-Not (Test-Path $distDir)) {
    throw "CARLA Python API distribution not found at $distDir"
  }

  $package = Get-ChildItem -Path $distDir -Include *.whl, *.egg | Select-Object -First 1
  if (-Not $package) {
    throw "Unable to locate CARLA wheel/egg in $distDir"
  }

  $pip = Join-Path $VenvPath "Scripts\pip.exe"
  Write-Host "Installing CARLA Python package from $($package.Name)"
  & $pip install $package.FullName

  $additionalPaths = @(
    (Join-Path $pythonApi "carla\lib"),
    (Join-Path $pythonApi "util"),
    (Join-Path $pythonApi "agents")
  ) | Where-Object { Test-Path $_ }

  if ($additionalPaths.Count -gt 0) {
    $pthPath = Join-Path $VenvPath "Lib\site-packages\carla_paths.pth"
    $additionalPaths | Set-Content -Path $pthPath
  }
}

function Configure-CarlaEnvironment {
  param(
    [string]$InstallRoot,
    [string]$Version
  )

  $carlaRoot = Join-Path $InstallRoot "CARLA_$Version"
  $carlaRoot = [System.IO.Path]::GetFullPath($carlaRoot)
  [Environment]::SetEnvironmentVariable("CARLA_ROOT", $carlaRoot, "User")
  Write-Host "CARLA_ROOT set to $carlaRoot"
}

$pythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Path
if (-Not $pythonPath) {
  $targetDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$($PythonVersion.Replace('.', ''))"
  Install-Python -Version $PythonVersion -Destination $targetDir
  $pythonPath = Join-Path $targetDir "python.exe"
}

$venv = Ensure-Venv -PythonExe $pythonPath
Install-PythonPackages -VenvPath $venv -UseCUDA:$InstallCUDA

$resolvedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$carlaInstallPath = Join-Path $resolvedInstallDir "CARLA_$CarlaVersion"
if (-Not (Test-Path $carlaInstallPath)) {
  Install-CARLA -Version $CarlaVersion -Destination $resolvedInstallDir
}

Install-CarlaPythonModule -VenvPath $venv -InstallRoot $resolvedInstallDir -Version $CarlaVersion
Configure-CarlaEnvironment -InstallRoot $resolvedInstallDir -Version $CarlaVersion

Write-Host "Environment setup complete. Activate with:`n`""$venv\Scripts\Activate.ps1""`
