Param(
  [string]$CarlaPath = "$PSScriptRoot/../../..",
  [int]$Port = 2000,
  [switch]$Offscreen
)

$ErrorActionPreference = "Stop"
$carlaExe = Join-Path $CarlaPath "CARLA_0.10.0\CarlaUE4.exe"
if (-Not (Test-Path $carlaExe)) {
  throw "CARLA executable not found at $carlaExe"
}

$arguments = @("-RenderOffScreen", "-quality-level=Epic", "-ResX=1920", "-ResY=1080", "-world-port=$Port")
if (-Not $Offscreen) {
  $arguments = @("-quality-level=Epic", "-world-port=$Port")
}

Start-Process -FilePath $carlaExe -ArgumentList $arguments -NoNewWindow
Write-Host "CARLA launched on port $Port"
