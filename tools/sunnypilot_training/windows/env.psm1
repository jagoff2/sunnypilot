function Enter-TrainingEnv {
  param(
    [string]$RepositoryRoot = "$PSScriptRoot/../../.."
  )
  $venv = Join-Path $RepositoryRoot "venv\Scripts\Activate.ps1"
  if (-Not (Test-Path $venv)) {
    throw "Virtual environment not found at $venv. Run setup_env.ps1 first."
  }
  & $venv
}

function Invoke-Carla {
  param(
    [int]$Port = 2000
  )
  $script = Join-Path $PSScriptRoot "start_carla.ps1"
  & $script -Port $Port -Offscreen
}

function Invoke-Trainer {
  param(
    [string]$Config = "default",
    [string]$OutputDir = "artifacts"
  )
  $repoRoot = Resolve-Path "$PSScriptRoot/../../.."
  $trainScript = Join-Path $repoRoot "tools/sunnypilot_training/training/train.py"
  python $trainScript --config-name $Config output_dir=$OutputDir
}

function Invoke-Collector {
  param(
    [string]$Scenario = "urban_intersection",
    [string]$Output = "data/zarr/train",
    [int]$Episodes = 10
  )
  $repoRoot = Resolve-Path "$PSScriptRoot/../../.."
  $collector = Join-Path $repoRoot "tools/sunnypilot_training/collector/collect_data.py"
  python $collector --scenario $Scenario --episodes $Episodes --output $Output
}

Export-ModuleMember -Function Enter-TrainingEnv, Invoke-Carla, Invoke-Trainer, Invoke-Collector
