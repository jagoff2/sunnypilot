Param(
  [string]$CarlaPath,
  [int]$Port = 2000,
  [switch]$Offscreen
)

$ErrorActionPreference = "Stop"

if (-Not $CarlaPath) {
  $envRoot = [Environment]::GetEnvironmentVariable("CARLA_ROOT", "User")
  if (-Not $envRoot) {
    $CarlaPath = "$PSScriptRoot/../../..\CARLA_0.10.0"
  }
  else {
    $CarlaPath = $envRoot
  }
}

$carlaExe = Join-Path $CarlaPath "CarlaUE4.exe"
if (-Not (Test-Path $carlaExe)) {
  throw "CARLA executable not found at $carlaExe"
}

$arguments = @("-RenderOffScreen", "-quality-level=Epic", "-ResX=1920", "-ResY=1080", "-world-port=$Port")
if (-Not $Offscreen) {
  $arguments = @("-quality-level=Epic", "-world-port=$Port")
}

Start-Process -FilePath $carlaExe -ArgumentList $arguments -NoNewWindow
Write-Host "CARLA launched on port $Port"
