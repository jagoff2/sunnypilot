Param(
  [string]$PythonVersion = "3.11.9",
  [switch]$InstallCUDA,
  [string]$CarlaVersion = "0.10.0",
  [string]$InstallDir = "$PSScriptRoot/../../.."
)

$ErrorActionPreference = "Stop"

Set-StrictMode -Version Latest

function Install-Python {
  param(
    [string]$Version,
    [string]$Destination
  )

  Write-Host "Installing Python $Version"
  $pythonInstaller = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
  $installerPath = Join-Path $env:TEMP "python-$Version.exe"
  Invoke-WebRequest -Uri $pythonInstaller -OutFile $installerPath -MaximumRedirection 5
  Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 TargetDir=$Destination" -Wait
  Remove-Item $installerPath -ErrorAction SilentlyContinue
}

function Resolve-Python {
  param(
    [string]$Version
  )

  $desiredMajorMinor = ($Version -split '\.')[0..1] -join '.'
  $targetDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$($Version.Replace('.', ''))"

  $existing = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($existing) {
    $existingPath = $null
    foreach ($property in 'Path', 'Source', 'Definition') {
      if ($existing.PSObject.Properties.Match($property).Count -gt 0) {
        $value = $existing.$property
        if ($value) {
          $existingPath = $value
          break
        }
      }
    }

    if ($existingPath) {
      $currentVersion = & $existingPath -c "import platform; print(platform.python_version())"
    }
    else {
      $currentVersion = $null
    }
    if ($null -eq $currentVersion -or -not $currentVersion.StartsWith($desiredMajorMinor)) {
      Install-Python -Version $Version -Destination $targetDir
      return (Join-Path $targetDir "python.exe")
    }
    else {
      Write-Host "Found Python $currentVersion on PATH"
      return $existingPath
    }
  }

  Install-Python -Version $Version -Destination $targetDir
  return (Join-Path $targetDir "python.exe")
}

function Ensure-Venv {
  param(
    [string]$PythonExe,
    [string]$ExpectedVersion
  )

  $venvPath = Join-Path $PSScriptRoot "..\..\..\venv"
  $venvPython = Join-Path $venvPath "Scripts\python.exe"
  $expectedPrefix = ($ExpectedVersion -split '\.')[0..1] -join '.'

  if (Test-Path $venvPython) {
    $venvVersion = & $venvPython -c "import platform; print(platform.python_version())"
    if (-not $venvVersion.StartsWith($expectedPrefix)) {
      Write-Host "Existing virtual environment targets Python $venvVersion. Recreating for $ExpectedVersion."
      Remove-Item $venvPath -Recurse -Force
    }
  }

  if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $PythonExe -m venv $venvPath
  }
  return $venvPath
}

function Get-GpuInfo {
  $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
  if (-not $smi) {
    return $null
  }

  try {
    $name = (& $smi.Path --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    $cap = (& $smi.Path --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
  }
  catch {
    return $null
  }

  if (-not $cap) {
    return $null
  }

  $culture = [System.Globalization.CultureInfo]::InvariantCulture
  $capability = 0.0
  if ([double]::TryParse($cap, [System.Globalization.NumberStyles]::Float, $culture, [ref]$capability)) {
    return [PSCustomObject]@{
      Name = $name
      ComputeCapability = $capability
    }
  }

  return $null
}

function Install-TorchPackages {
  param(
    [string]$PythonExe,
    [switch]$UseCUDA
  )

  $pipArgs = @('-m', 'pip', 'install', '--no-cache-dir')
  $culture = [System.Globalization.CultureInfo]::InvariantCulture

  if ($UseCUDA) {
    $gpuInfo = Get-GpuInfo
    $capability = $null
    if ($gpuInfo) {
      $capability = $gpuInfo.ComputeCapability
      $formattedCapability = $capability.ToString('0.0', $culture)
      Write-Host "Detected GPU: $($gpuInfo.Name) (compute capability $formattedCapability)"
      if ($gpuInfo.Name -match '5060') {
        Write-Host "Verified RTX 5060 compute capability $formattedCapability"
      }
    }
    else {
      Write-Warning "Unable to query NVIDIA GPU details. Proceeding with CUDA-enabled PyTorch installation."
    }

    if ($capability -and $capability -lt 9.0) {
      $torchArgs = @(
        'torch==2.4.1+cu121',
        'torchvision==0.19.1+cu121',
        'torchaudio==2.4.1',
        '--index-url', 'https://download.pytorch.org/whl/cu121'
      )
      Write-Host "Installing PyTorch 2.4.1 with CUDA 12.1 support"
    }
    else {
      $torchArgs = @(
        'torch==2.5.1+cu124',
        'torchvision==0.20.1+cu124',
        'torchaudio==2.5.1',
        '--index-url', 'https://download.pytorch.org/whl/cu124'
      )
      Write-Host "Installing PyTorch 2.5.1 with CUDA 12.4 support"
    }

    & $PythonExe @($pipArgs + $torchArgs)
  }
  else {
    $torchArgs = @(
      'torch==2.5.1',
      'torchvision==0.20.1',
      'torchaudio==2.5.1'
    )
    Write-Host "Installing CPU PyTorch 2.5.1"
    & $PythonExe @($pipArgs + $torchArgs)
  }
}

function Install-PythonPackages {
  param(
    [string]$VenvPath,
    [switch]$UseCUDA
  )

  $python = Join-Path $VenvPath "Scripts\python.exe"
  $env:PIP_NO_CACHE_DIR = '1'
  & $python -m pip install --upgrade pip
  & $python -m pip install --upgrade setuptools wheel

  Install-TorchPackages -PythonExe $python -UseCUDA:$UseCUDA

  & $python -m pip install --no-cache-dir hydra-core==1.3.2 omegaconf==2.3.0 tinygrad==0.8.0 zarr numpy scipy tqdm pygame pytest
  Remove-Item Env:PIP_NO_CACHE_DIR -ErrorAction SilentlyContinue
}

function Install-CARLA {
  param(
    [string]$Version,
    [string]$Destination
  )

  $token = $Version.Replace('.', '-')
  $carlaZip = "https://tiny.carla.org/carla-$token-windows"
  $downloadPath = Join-Path $env:TEMP "CARLA_$Version.zip"
  Write-Host "Downloading CARLA $Version"
  Invoke-WebRequest -Uri $carlaZip -OutFile $downloadPath -MaximumRedirection 5
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

  $python = Join-Path $VenvPath "Scripts\python.exe"
  Write-Host "Installing CARLA Python package from $($package.Name)"
  & $python -m pip install --no-cache-dir $package.FullName

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

$pythonPath = Resolve-Python -Version $PythonVersion

$venv = Ensure-Venv -PythonExe $pythonPath -ExpectedVersion $PythonVersion
Install-PythonPackages -VenvPath $venv -UseCUDA:$InstallCUDA

$resolvedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$carlaInstallPath = Join-Path $resolvedInstallDir "CARLA_$CarlaVersion"
if (-Not (Test-Path $carlaInstallPath)) {
  Install-CARLA -Version $CarlaVersion -Destination $resolvedInstallDir
}

Install-CarlaPythonModule -VenvPath $venv -InstallRoot $resolvedInstallDir -Version $CarlaVersion
Configure-CarlaEnvironment -InstallRoot $resolvedInstallDir -Version $CarlaVersion

Write-Host "Environment setup complete. Activate with:`n`""$venv\Scripts\Activate.ps1""`
