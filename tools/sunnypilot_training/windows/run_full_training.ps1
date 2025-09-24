Param(
  [string]$Scenario = "urban_intersection",
  [ValidateRange(0, 1000000)]
  [int]$TrainEpisodes = 50,
  [ValidateRange(0, 1000000)]
  [int]$ValEpisodes = 10,
  [string]$DatasetRoot = "$PSScriptRoot/../../data/zarr",
  [string]$OutputDir = "$PSScriptRoot/../artifacts",
  [string]$ConfigName = "default",
  [int]$MaxEpochs,
  [int]$CheckpointInterval,
  [string[]]$ConfigOverride = @(),
  [string]$CheckpointPath,
  [string]$ExportDir,
  [switch]$SkipDataCollection,
  [switch]$SkipTraining,
  [switch]$SkipExport,
  [switch]$InstallModels,
  [string]$OpenpilotRoot,
  [int]$CarlaPort = 2000,
  [string]$CarlaHost = "127.0.0.1",
  [int]$CarlaReadyTimeout = 120,
  [switch]$KeepCarlaRunning,
  [int]$TrainShardSize = 512,
  [int]$ValShardSize = 512,
  [int]$CollectorSeed
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-PythonExe {
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if (-not $pythonCmd) {
    throw "python executable not found on PATH. Run Enter-TrainingEnv from setup_env.ps1 first."
  }
  if ($pythonCmd.Path) {
    return $pythonCmd.Path
  }
  if ($pythonCmd.Source) {
    return $pythonCmd.Source
  }
  throw "Unable to resolve python executable path from $pythonCmd."
}

function Format-PathForHydra {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $posixPath = $fullPath -replace '\\', '/'
  $escaped = $posixPath -replace '"', '\"'
  return [string]::Concat('"', $escaped, '"')
}

function Ensure-Directory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function Invoke-PythonCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Description,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments,
    [string]$PythonExe
  )
  Write-Host "==> $Description"
  & $PythonExe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to $Description (exit code $LASTEXITCODE). Review the output above for details."
  }
}

function Invoke-PythonModule {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Description,
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [string[]]$Arguments = @(),
    [string]$PythonExe
  )
  $moduleArgs = @("-m", $Module)
  if ($Arguments) {
    $moduleArgs += $Arguments
  }
  Invoke-PythonCommand -Description $Description -Arguments $moduleArgs -PythonExe $PythonExe
}

function Start-CarlaProcess {
  param(
    [int]$Port,
    [string]$RepoRoot
  )
  $processNames = @("CarlaUnreal", "CarlaUE4")
  $existing = foreach ($name in $processNames) {
    Get-Process -Name $name -ErrorAction SilentlyContinue
  }
  $existing = $existing | Where-Object { $_ }
  if ($existing) {
    $names = ($existing | Select-Object -ExpandProperty ProcessName -Unique) -join ', '
    $pids = ($existing | Select-Object -ExpandProperty Id) -join ', '
    throw "Detected an existing CARLA process ($names) with PID(s): $pids. Stop it before running the automated pipeline."
  }
  $carlaRoot = $null
  foreach ($scope in @("Process", "User", "Machine")) {
    $value = [Environment]::GetEnvironmentVariable("CARLA_ROOT", $scope)
    if ($value) {
      $carlaRoot = $value
      break
    }
  }
  if (-not $carlaRoot) {
    $carlaRoot = Join-Path $RepoRoot "CARLA_0.10.0"
  }
  $carlaRoot = [System.IO.Path]::GetFullPath($carlaRoot)
  $carlaExe = $null
  $searchBases = @()
  if (Test-Path $carlaRoot -PathType Leaf) {
    if ($carlaRoot.ToLower().EndsWith('.exe')) {
      $carlaExe = $carlaRoot
    }
    $searchBases = @([System.IO.Path]::GetDirectoryName($carlaRoot))
  }

  $candidateExeNames = @(
    "CarlaUnreal.exe",
    "CarlaUE4.exe",
    "CarlaUnreal-Win64-Shipping.exe",
    "CarlaUE4-Win64-Shipping.exe"
  )
  $relativeSearchPaths = @(
    "",
    "WindowsNoEditor",
    "CarlaUnreal",
    "CarlaUE4",
    "CarlaUnreal\\Binaries\\Win64",
    "CarlaUE4\\Binaries\\Win64",
    "WindowsNoEditor\\CarlaUnreal\\Binaries\\Win64",
    "WindowsNoEditor\\CarlaUE4\\Binaries\\Win64"
  )

  if (-not $carlaExe) {
    if (-not $searchBases) {
      $searchBases = @($carlaRoot, (Join-Path $carlaRoot "WindowsNoEditor"))
    }

    $candidateBases = New-Object System.Collections.Generic.List[string]
    foreach ($base in $searchBases | Where-Object { $_ }) {
      foreach ($relative in $relativeSearchPaths) {
        if ([string]::IsNullOrWhiteSpace($relative)) {
          $candidateBase = $base
        }
        else {
          $candidateBase = Join-Path $base $relative
        }

        if ($candidateBase -and -not $candidateBases.Contains($candidateBase)) {
          $candidateBases.Add($candidateBase)
        }
      }
    }

    foreach ($base in $candidateBases) {
      if (-not (Test-Path $base)) {
        continue
      }
      foreach ($exeName in $candidateExeNames) {
        $candidate = Join-Path $base $exeName
        if (Test-Path $candidate) {
          $carlaExe = $candidate
          break
        }
      }
      if ($carlaExe) { break }
    }

    if (-not $carlaExe) {
      $expectedPaths = foreach ($base in $candidateBases) {
        foreach ($exeName in $candidateExeNames) {
          Join-Path $base $exeName
        }
      }
      if ($carlaRoot.ToLower().EndsWith('.exe')) {
        $expectedPaths += $carlaRoot
      }
      $expectedList = ($expectedPaths) -join ', '
      throw "CARLA executable not found. Expected one of: $expectedList. Re-run setup_env.ps1 with the default CARLA installation."
    }
  }
  $carlaWorkingDir = [System.IO.Path]::GetDirectoryName($carlaExe)
  $arguments = @("-RenderOffScreen", "-quality-level=Epic", "-ResX=1920", "-ResY=1080", "-carla-rpc-port=$Port")
  $process = Start-Process -FilePath $carlaExe -ArgumentList $arguments -PassThru -WorkingDirectory $carlaWorkingDir -WindowStyle Hidden
  Write-Host "Launched CARLA (PID $($process.Id)) on port $Port"
  return $process
}

function Test-CarlaReady {
  param(
    [string]$PythonExe,
    [string]$CarlaHost,
    [int]$Port
  )
  $scriptTemplate = @'
import sys
import traceback

try:
  import carla
except ModuleNotFoundError as exc:
  print(f"MODULE_NOT_FOUND:{exc}")
  sys.exit(2)
except Exception:
  traceback.print_exc()
  sys.exit(3)

client = carla.Client("{{CARLA_HOST}}", {{CARLA_PORT}})
client.set_timeout(2.0)

try:
  client.get_world()
except Exception as exc:
  print(f"CONNECTION_ERROR:{exc}")
  sys.exit(1)
else:
  sys.exit(0)
'@
  $scriptContent = $scriptTemplate.Replace("{{CARLA_HOST}}", $CarlaHost).Replace("{{CARLA_PORT}}", [string]$Port)
  $tempScriptPath = [System.IO.Path]::ChangeExtension(
    [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName()),
    ".py"
  )
  try {
    [System.IO.File]::WriteAllText($tempScriptPath, $scriptContent)
    $output = & $PythonExe $tempScriptPath 2>&1
    $exitCode = $LASTEXITCODE
    $joinedOutput = if ($output) { ($output -join [Environment]::NewLine).Trim() } else { "" }
    return [PSCustomObject]@{
      Success = ($exitCode -eq 0)
      ExitCode = $exitCode
      Output = $joinedOutput
    }
  }
  finally {
    if (Test-Path $tempScriptPath) {
      Remove-Item -Path $tempScriptPath -Force
    }
  }
}

function Wait-CarlaReady {
  param(
    [string]$PythonExe,
    [string]$CarlaHost,
    [int]$Port,
    [int]$Timeout
  )
  $deadline = (Get-Date).AddSeconds($Timeout)
  $connectionMessagePrinted = $false
  while ((Get-Date) -lt $deadline) {
    $result = Test-CarlaReady -PythonExe $PythonExe -CarlaHost $CarlaHost -Port $Port
    if ($result.Success) {
      Write-Host "CARLA is ready on ${CarlaHost}:${Port}"
      return
    }
    if ($result.ExitCode -eq 2) {
      if ($result.Output) {
        Write-Host $result.Output
      }
      throw "Python module 'carla' is not available for $PythonExe. Activate the training environment (Import-Module .\\env.psm1; Enter-TrainingEnv) or rerun setup_env.ps1."
    }
    if ($result.ExitCode -eq 3) {
      if ($result.Output) {
        Write-Host $result.Output
      }
      throw "Failed to import the CARLA Python API with $PythonExe. Review the traceback above and reinstall the environment if needed."
    }
    if (-not $connectionMessagePrinted) {
      if ($result.Output) {
        Write-Host $result.Output
      }
      Write-Host "Waiting for CARLA to accept connections on ${CarlaHost}:${Port}..."
      $connectionMessagePrinted = $true
    }
    Start-Sleep -Seconds 2
  }
  throw "Timed out waiting for CARLA to respond on ${CarlaHost}:${Port} after $Timeout seconds."
}

function Ensure-ShardsPresent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Directory,
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [int]$ExpectedEpisodes
  )
  $shards = Get-ChildItem -Path $Directory -Filter '*.zarr' -ErrorAction SilentlyContinue
  if (-not $shards) {
    throw "No .zarr shards were produced for $Label data in $Directory."
  }
  if ($ExpectedEpisodes -gt 0 -and $shards.Count -lt $ExpectedEpisodes) {
    throw "Expected $ExpectedEpisodes $Label shards in $Directory but found $($shards.Count)."
  }
  Write-Host "Found $($shards.Count) $Label shards in $Directory"
}

function Get-LatestCheckpointPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointDir
  )
  if (-not (Test-Path $CheckpointDir)) {
    throw "Checkpoint directory not found: $CheckpointDir"
  }
  $checkpoints = Get-ChildItem -Path $CheckpointDir -Filter 'epoch_*.pt' | Sort-Object Name
  if (-not $checkpoints) {
    throw "No training checkpoints found in $CheckpointDir"
  }
  return $checkpoints[-1].FullName
}

$pythonExe = Resolve-PythonExe
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../.."))
$toolsRoot = Join-Path $repoRoot "tools/sunnypilot_training"
$collectorModule = "tools.sunnypilot_training.collector.collect_data"
$trainModule = "tools.sunnypilot_training.training.train"
$visionExportModule = "tools.sunnypilot_training.export.export_vision"
$policyExportModule = "tools.sunnypilot_training.export.export_policy"
$integrationScript = Join-Path $toolsRoot "integration/replace_models.ps1"

if (-not (Test-Path $toolsRoot)) {
  throw "Unable to locate sunnypilot training tools at $toolsRoot"
}

$pathSeparator = [System.IO.Path]::PathSeparator
if ($env:PYTHONPATH) {
  $paths = $env:PYTHONPATH -split [System.Text.RegularExpressions.Regex]::Escape($pathSeparator)
  if ($paths -notcontains $repoRoot) {
    $env:PYTHONPATH = "$repoRoot$pathSeparator$env:PYTHONPATH"
  }
}
else {
  $env:PYTHONPATH = $repoRoot
}

$datasetRootFull = [System.IO.Path]::GetFullPath($DatasetRoot)
$trainDir = Join-Path $datasetRootFull "train"
$valDir = Join-Path $datasetRootFull "val"
$trainingOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if (-not $ExportDir) {
  $exportRoot = $trainingOutputDir
}
else {
  $exportRoot = [System.IO.Path]::GetFullPath($ExportDir)
}
if (-not $OpenpilotRoot) {
  $resolvedOpenpilotRoot = $repoRoot
}
else {
  $resolvedOpenpilotRoot = [System.IO.Path]::GetFullPath($OpenpilotRoot)
}

Ensure-Directory $datasetRootFull
Ensure-Directory $trainDir
Ensure-Directory $valDir
Ensure-Directory $trainingOutputDir
Ensure-Directory $exportRoot

$carlaProcess = $null
$originalLocation = Get-Location
Push-Location $repoRoot
try {
  if (-not $SkipDataCollection) {
    $carlaProcess = Start-CarlaProcess -Port $CarlaPort -RepoRoot $repoRoot
    Wait-CarlaReady -PythonExe $pythonExe -CarlaHost $CarlaHost -Port $CarlaPort -Timeout $CarlaReadyTimeout

    if ($TrainEpisodes -gt 0) {
      $trainCollectorArgs = @(
        "--host", $CarlaHost,
        "--port", $CarlaPort.ToString(),
        "--scenario", $Scenario,
        "--episodes", $TrainEpisodes.ToString(),
        "--output", $trainDir,
        "--shard-size", $TrainShardSize.ToString()
      )
      if ($PSBoundParameters.ContainsKey('CollectorSeed')) {
        $trainCollectorArgs += @("--seed", $CollectorSeed.ToString())
      }
      Invoke-PythonModule -Description "collect $TrainEpisodes training episodes" -Module $collectorModule -Arguments $trainCollectorArgs -PythonExe $pythonExe
      Ensure-ShardsPresent -Directory $trainDir -Label "training" -ExpectedEpisodes $TrainEpisodes
    }
    else {
      Write-Host "Skipping training data collection because TrainEpisodes is 0."
    }

    if ($ValEpisodes -gt 0) {
      $valCollectorArgs = @(
        "--host", $CarlaHost,
        "--port", $CarlaPort.ToString(),
        "--scenario", $Scenario,
        "--episodes", $ValEpisodes.ToString(),
        "--output", $valDir,
        "--shard-size", $ValShardSize.ToString()
      )
      if ($PSBoundParameters.ContainsKey('CollectorSeed')) {
        $valCollectorArgs += @("--seed", ($CollectorSeed + 1).ToString())
      }
      Invoke-PythonModule -Description "collect $ValEpisodes validation episodes" -Module $collectorModule -Arguments $valCollectorArgs -PythonExe $pythonExe
      Ensure-ShardsPresent -Directory $valDir -Label "validation" -ExpectedEpisodes $ValEpisodes
    }
    else {
      Write-Host "Skipping validation data collection because ValEpisodes is 0."
    }

    if (-not $KeepCarlaRunning -and $carlaProcess) {
      if (-not $carlaProcess.HasExited) {
        Write-Host "Stopping CARLA (PID $($carlaProcess.Id))"
        Stop-Process -Id $carlaProcess.Id -Force
      }
      $carlaProcess = $null
    }
  }
  else {
    Write-Host "Skipping data collection as requested."
    if ($TrainEpisodes -gt 0) {
      Ensure-ShardsPresent -Directory $trainDir -Label "training" -ExpectedEpisodes 1
    }
    if ($ValEpisodes -gt 0) {
      Ensure-ShardsPresent -Directory $valDir -Label "validation" -ExpectedEpisodes 1
    }
  }

  if (-not $SkipTraining) {
    if ($TrainEpisodes -gt 0) {
      Ensure-ShardsPresent -Directory $trainDir -Label "training" -ExpectedEpisodes 1
    }
    if ($ValEpisodes -gt 0) {
      Ensure-ShardsPresent -Directory $valDir -Label "validation" -ExpectedEpisodes 1
    }
    $overrides = @(
      "dataset.train_path=$(Format-PathForHydra $trainDir)",
      "dataset.val_path=$(Format-PathForHydra $valDir)",
      "output_dir=$(Format-PathForHydra $trainingOutputDir)"
    )
    if ($PSBoundParameters.ContainsKey('MaxEpochs')) {
      $overrides += "max_epochs=$MaxEpochs"
    }
    if ($PSBoundParameters.ContainsKey('CheckpointInterval')) {
      $overrides += "checkpoint_interval=$CheckpointInterval"
    }
    if ($ConfigOverride) {
      $overrides += $ConfigOverride
    }
    $trainArgs = @(
      "--config-name", $ConfigName
    ) + $overrides
    Invoke-PythonModule -Description "train diffusion models" -Module $trainModule -Arguments $trainArgs -PythonExe $pythonExe
  }
  else {
    Write-Host "Skipping training as requested."
  }

  $checkpointToExport = $null
  if ($PSBoundParameters.ContainsKey('CheckpointPath')) {
    if (-not (Test-Path $CheckpointPath)) {
      throw "Specified checkpoint path not found: $CheckpointPath"
    }
    $checkpointToExport = [System.IO.Path]::GetFullPath($CheckpointPath)
  }
  else {
    $checkpointDir = Join-Path $trainingOutputDir "checkpoints"
    $checkpointToExport = Get-LatestCheckpointPath -CheckpointDir $checkpointDir
  }
  Write-Host "Using checkpoint $checkpointToExport for export"

  $visionOnnx = Join-Path $exportRoot "driving_vision.onnx"
  $visionTinygrad = Join-Path $exportRoot "driving_vision_tinygrad.pkl"
  $visionMetadata = Join-Path $exportRoot "driving_vision_metadata.pkl"
  $policyOnnx = Join-Path $exportRoot "driving_policy.onnx"
  $policyTinygrad = Join-Path $exportRoot "driving_policy_tinygrad.pkl"
  $policyMetadata = Join-Path $exportRoot "driving_policy_metadata.pkl"

  if (-not $SkipExport) {
    $visionArgs = @(
      $checkpointToExport,
      "--onnx", $visionOnnx,
      "--tinygrad", $visionTinygrad,
      "--metadata", $visionMetadata
    )
    Invoke-PythonModule -Description "export vision model" -Module $visionExportModule -Arguments $visionArgs -PythonExe $pythonExe

    $policyArgs = @(
      $checkpointToExport,
      "--onnx", $policyOnnx,
      "--tinygrad", $policyTinygrad,
      "--metadata", $policyMetadata
    )
    Invoke-PythonModule -Description "export policy model" -Module $policyExportModule -Arguments $policyArgs -PythonExe $pythonExe
  }
  else {
    Write-Host "Skipping export as requested."
  }

  if ($InstallModels) {
    Write-Host "Installing exported models into $resolvedOpenpilotRoot"
    & $integrationScript `
      -VisionOnnx $visionOnnx `
      -VisionTinygrad $visionTinygrad `
      -VisionMetadata $visionMetadata `
      -PolicyOnnx $policyOnnx `
      -PolicyTinygrad $policyTinygrad `
      -PolicyMetadata $policyMetadata `
      -OpenpilotRoot $resolvedOpenpilotRoot
    if ($LASTEXITCODE -ne 0) {
      throw "Model installation script reported exit code $LASTEXITCODE."
    }
  }

  Write-Host "`nTraining pipeline completed successfully. Final artifacts:"
  Write-Host "  Vision ONNX: $visionOnnx"
  Write-Host "  Vision tinygrad: $visionTinygrad"
  Write-Host "  Vision metadata: $visionMetadata"
  Write-Host "  Policy ONNX: $policyOnnx"
  Write-Host "  Policy tinygrad: $policyTinygrad"
  Write-Host "  Policy metadata: $policyMetadata"

  [PSCustomObject]@{
    Checkpoint = $checkpointToExport
    VisionOnnx = $visionOnnx
    VisionTinygrad = $visionTinygrad
    VisionMetadata = $visionMetadata
    PolicyOnnx = $policyOnnx
    PolicyTinygrad = $policyTinygrad
    PolicyMetadata = $policyMetadata
  }
}
catch {
  $message = $_.Exception.Message
  if ($_.InvocationInfo) {
    $message = "$message (at $($_.InvocationInfo.PositionMessage.Trim()))"
  }
  throw "Training pipeline failed: $message"
}
finally {
  if (-not $KeepCarlaRunning -and $carlaProcess -and -not $carlaProcess.HasExited) {
    try {
      Write-Host "Stopping CARLA (PID $($carlaProcess.Id))"
      Stop-Process -Id $carlaProcess.Id -Force
    }
    catch {
      Write-Warning "Failed to stop CARLA process cleanly: $($_.Exception.Message)"
    }
  }
  Pop-Location
  Set-Location $originalLocation.Path
}
