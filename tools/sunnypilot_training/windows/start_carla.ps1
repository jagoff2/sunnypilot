Param(
  [string]$CarlaPath,
  [int]$Port = 2000,
  [switch]$Offscreen
)

$ErrorActionPreference = "Stop"

if (-Not $CarlaPath) {
  foreach ($scope in @(
      "Process",
      "User",
      "Machine"
    )) {
    $value = [Environment]::GetEnvironmentVariable("CARLA_ROOT", $scope)
    if ($value) {
      $CarlaPath = $value
      break
    }
  }
  if (-Not $CarlaPath) {
    $CarlaPath = "$PSScriptRoot/../../..\CARLA_0.10.0"
  }
}

$CarlaPath = [System.IO.Path]::GetFullPath($CarlaPath)
$carlaExe = $null
$searchBases = @()

if (Test-Path $CarlaPath -PathType Leaf) {
  if ($CarlaPath.ToLower().EndsWith('.exe')) {
    $carlaExe = $CarlaPath
  }
  $searchBases = @([System.IO.Path]::GetDirectoryName($CarlaPath))
}

if (-Not $carlaExe) {
  if (-not $searchBases) {
    $searchBases = @($CarlaPath, (Join-Path $CarlaPath "WindowsNoEditor"))
  }
  foreach ($base in $searchBases | Where-Object { $_ }) {
    foreach ($exeName in @(
        "CarlaUnreal.exe",
        "CarlaUE4.exe"
      )) {
      $candidate = Join-Path $base $exeName
      if (Test-Path $candidate) {
        $carlaExe = $candidate
        break
      }
    }
    if ($carlaExe) { break }
  }
}

if (-Not $carlaExe) {
  $expected = foreach ($base in $searchBases | Where-Object { $_ }) {
    foreach ($exeName in @(
        "CarlaUnreal.exe",
        "CarlaUE4.exe"
      )) {
      Join-Path $base $exeName
    }
  }
  if ($CarlaPath.ToLower().EndsWith('.exe')) {
    $expected += $CarlaPath
  }
  $expectedList = ($expected) -join ', '
  throw "CARLA executable not found. Expected: $expectedList"
}

$arguments = @(
  "-RenderOffScreen",
  "-quality-level=Epic",
  "-ResX=1920",
  "-ResY=1080",
  "-carla-rpc-port=$Port"
)
if (-Not $Offscreen) {
  $arguments = @(
    "-quality-level=Epic",
    "-carla-rpc-port=$Port"
  )
}

$workingDir = Split-Path -Path $carlaExe -Parent
Start-Process -FilePath $carlaExe -ArgumentList $arguments -NoNewWindow -WorkingDirectory $workingDir
Write-Host "CARLA launched on port $Port"
