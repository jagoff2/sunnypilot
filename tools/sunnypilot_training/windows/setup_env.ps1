Param(
  [string]$PythonVersion = "3.11.9",
  [switch]$InstallCUDA,
  [string]$CarlaVersion = "0.10.0",
  [string]$InstallDir = "$PSScriptRoot/../../.."
)

$ErrorActionPreference = "Stop"

Set-StrictMode -Version Latest

function Get-MajorMinorVersionString {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$ParameterName = "Version"
  )

  $parts = $Version -split '\.'
  if ($parts.Length -ge 2) {
    return "$($parts[0]).$($parts[1])"
  }

  throw "Unable to determine major.minor from $ParameterName '$Version'. Provide a value like '3.11.9'."
}

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

  $desiredMajorMinor = Get-MajorMinorVersionString -Version $Version -ParameterName "Python version"
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
  $expectedPrefix = Get-MajorMinorVersionString -Version $ExpectedVersion -ParameterName "virtual environment Python version"

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

function Find-CarlaPythonPackage {
  param(
    [string]$CarlaRoot
  )

  $pythonApi = Join-Path $CarlaRoot "PythonAPI"
  if (-Not (Test-Path $pythonApi)) {
    return $null
  }

  $candidateDirs = @(
    Join-Path $pythonApi "carla\dist",
    Join-Path $pythonApi "dist"
  )

  foreach ($dir in $candidateDirs) {
    if (Test-Path $dir) {
      $package = Get-ChildItem -Path $dir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @('.whl', '.egg') } |
        Select-Object -First 1
      if ($package) {
        return [PSCustomObject]@{
          Package = $package
          DistDirectory = $dir
          PythonApi = $pythonApi
        }
      }
    }
  }

  $fallbackPackage = Get-ChildItem -Path $pythonApi -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @('.whl', '.egg') } |
    Select-Object -First 1

  if ($fallbackPackage) {
    return [PSCustomObject]@{
      Package = $fallbackPackage
      DistDirectory = $fallbackPackage.DirectoryName
      PythonApi = $pythonApi
    }
  }

  return $null
}

function Resolve-CarlaInstallPath {
  param(
    [string]$InstallRoot,
    [string]$Version
  )

  if (-not (Test-Path $InstallRoot)) {
    return $null
  }

  $versions = @($Version, $Version.Replace('.', '_'))
  $candidateNames = @()

  foreach ($variant in $versions) {
    $candidateNames += @(
      "CARLA_$variant",
      "CARLA-$variant",
      "CARLA-$variant-Win64-Shipping",
      "CARLA_$variant-Win64-Shipping",
      "Carla-$variant",
      "Carla-$variant-Win64-Shipping",
      "Carla_$variant",
      "Carla_$variant-Win64-Shipping"
    )
  }

  $candidateNames = $candidateNames | Select-Object -Unique

  foreach ($name in $candidateNames) {
    $candidatePath = Join-Path $InstallRoot $name
    if (Test-Path $candidatePath) {
      $package = Find-CarlaPythonPackage -CarlaRoot $candidatePath
      if ($package) {
        return $candidatePath
      }
    }
  }

  $directories = Get-ChildItem -Path $InstallRoot -Directory -ErrorAction SilentlyContinue
  foreach ($dir in $directories) {
    foreach ($versionString in $versions) {
      if ($dir.Name -like "*$versionString*") {
        $package = Find-CarlaPythonPackage -CarlaRoot $dir.FullName
        if ($package) {
          return $dir.FullName
        }
      }
    }
  }

  return $null
}

function Install-CARLA {
  param(
    [string]$Version,
    [string]$Destination
  )

  $targetDir = Join-Path $Destination "CARLA_$Version"
  $targetFullPath = [System.IO.Path]::GetFullPath($targetDir)

  $existing = Resolve-CarlaInstallPath -InstallRoot $Destination -Version $Version
  if ($existing) {
    $existingFullPath = [System.IO.Path]::GetFullPath($existing)
    if ($existingFullPath -ieq $targetFullPath) {
      Write-Host "CARLA $Version already installed at $targetDir"
      return $targetDir
    }

    Write-Host "Found existing CARLA $Version installation at $existing. Normalizing to $targetDir"
    if (Test-Path $targetDir) {
      Remove-Item $targetDir -Recurse -Force
    }
    Move-Item -Path $existing -Destination $targetDir
    return $targetDir
  }

  $token = $Version.Replace('.', '-')
  $carlaZip = "https://tiny.carla.org/carla-$token-windows"
  $downloadPath = Join-Path $env:TEMP "CARLA_$Version.zip"
  if (Test-Path $downloadPath) {
    Write-Host "Using existing CARLA $Version archive at $downloadPath"
  }
  else {
    Write-Host "Downloading CARLA $Version"
    Invoke-WebRequest -Uri $carlaZip -OutFile $downloadPath -MaximumRedirection 5
  }

  Write-Host "Extracting CARLA archive to $Destination"
  if (-Not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
  }
  Expand-Archive -Path $downloadPath -DestinationPath $Destination -Force
  Remove-Item $downloadPath -ErrorAction SilentlyContinue

  $extracted = Resolve-CarlaInstallPath -InstallRoot $Destination -Version $Version
  if (-not $extracted) {
    throw "CARLA Python API distribution not found after extracting archive to $Destination"
  }

  $extractedFullPath = [System.IO.Path]::GetFullPath($extracted)
  if ($extractedFullPath -ieq $targetFullPath) {
    return $targetDir
  }

  if (Test-Path $targetDir) {
    Remove-Item $targetDir -Recurse -Force
  }

  Move-Item -Path $extracted -Destination $targetDir
  return $targetDir
}

function Install-CarlaPythonModule {
  param(
    [string]$VenvPath,
    [string]$InstallRoot,
    [string]$Version,
    [string]$CarlaRoot = $null
  )

  if (-not $CarlaRoot) {
    $CarlaRoot = Resolve-CarlaInstallPath -InstallRoot $InstallRoot -Version $Version
  }
  if (-not $CarlaRoot) {
    throw "CARLA Python API distribution not found for version $Version under $InstallRoot"
  }

  $packageInfo = Find-CarlaPythonPackage -CarlaRoot $CarlaRoot
  if (-not $packageInfo) {
    throw "Unable to locate CARLA wheel/egg under $CarlaRoot"
  }

  $python = Join-Path $VenvPath "Scripts\python.exe"
  Write-Host "Installing CARLA Python package from $($packageInfo.Package.FullName)"
  & $python -m pip install --no-cache-dir $packageInfo.Package.FullName

  $pythonApi = $packageInfo.PythonApi
  $additionalPaths = @(
    (Join-Path (Join-Path $pythonApi 'carla') 'lib'),
    (Join-Path $pythonApi 'util'),
    (Join-Path $pythonApi 'agents')
  ) | Where-Object { Test-Path $_ }

  if ($additionalPaths.Count -gt 0) {
    $pthPath = Join-Path $VenvPath "Lib\site-packages\carla_paths.pth"
    $additionalPaths | Set-Content -Path $pthPath
  }
}

function Configure-CarlaEnvironment {
  param(
    [string]$InstallRoot,
    [string]$Version,
    [string]$CarlaRoot = $null
  )

  if (-not $CarlaRoot) {
    $CarlaRoot = Resolve-CarlaInstallPath -InstallRoot $InstallRoot -Version $Version
  }
  if (-not $CarlaRoot) {
    throw "CARLA installation for version $Version not found under $InstallRoot"
  }
  $resolvedCarlaRoot = [System.IO.Path]::GetFullPath($CarlaRoot)
  [Environment]::SetEnvironmentVariable("CARLA_ROOT", $resolvedCarlaRoot, "User")
  Write-Host "CARLA_ROOT set to $resolvedCarlaRoot"
}

$pythonPath = Resolve-Python -Version $PythonVersion

$venv = Ensure-Venv -PythonExe $pythonPath -ExpectedVersion $PythonVersion
Install-PythonPackages -VenvPath $venv -UseCUDA:$InstallCUDA

$resolvedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$carlaRoot = Install-CARLA -Version $CarlaVersion -Destination $resolvedInstallDir
if (-not $carlaRoot) {
  throw "Failed to locate CARLA installation for version $CarlaVersion"
}

Install-CarlaPythonModule -VenvPath $venv -InstallRoot $resolvedInstallDir -Version $CarlaVersion -CarlaRoot $carlaRoot
Configure-CarlaEnvironment -InstallRoot $resolvedInstallDir -Version $CarlaVersion -CarlaRoot $carlaRoot

Write-Host "Environment setup complete. Activate with:`n`""$venv\Scripts\Activate.ps1""`
