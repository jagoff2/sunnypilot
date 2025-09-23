Param(
  [string]$VisionOnnx,
  [string]$VisionTinygrad,
  [string]$VisionMetadata,
  [string]$PolicyOnnx,
  [string]$PolicyTinygrad,
  [string]$PolicyMetadata,
  [string]$OpenpilotRoot = "$PSScriptRoot/../../.."
)

$ErrorActionPreference = "Stop"
$modelsPath = Join-Path $OpenpilotRoot "selfdrive/modeld/models"
if (-Not (Test-Path $modelsPath)) {
  throw "Model directory not found: $modelsPath"
}

Copy-Item $VisionOnnx (Join-Path $modelsPath "driving_vision.onnx") -Force
Copy-Item $VisionTinygrad (Join-Path $modelsPath "driving_vision_tinygrad.pkl") -Force
Copy-Item $VisionMetadata (Join-Path $modelsPath "driving_vision_metadata.pkl") -Force
Copy-Item $PolicyOnnx (Join-Path $modelsPath "driving_policy.onnx") -Force
Copy-Item $PolicyTinygrad (Join-Path $modelsPath "driving_policy_tinygrad.pkl") -Force
Copy-Item $PolicyMetadata (Join-Path $modelsPath "driving_policy_metadata.pkl") -Force

Write-Host "Models updated in $modelsPath"
