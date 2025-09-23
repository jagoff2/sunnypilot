# Sunnypilot diffusion training suite

This package delivers an end-to-end workflow for training and exporting diffusion-based
vision and policy models that comply with the contracts expected by `selfdrive/modeld`.
It targets Windows hosts and CARLA 0.10.0.

## Components

- **Scenario orchestration** (`scenarios/`): deterministic definitions for intersections, highways,
  and four-way stops. The registry exposes a turnkey API for spawning seeded scenarios.
- **Sensor + data logging** (`collector/collect_data.py`): attaches center-mounted AR0231-equivalent
  cameras, preprocesses frames into the expected YUV stacks, and logs perception/policy targets.
- **Dataset utilities** (`dataset/`): chunked Zarr writer and PyTorch streaming dataset with
  photometric + temporal augmentations.
- **Models** (`models/`): DiffusionDrive-inspired vision encoder and diffusion-based policy head.
- **Training loop** (`training/train.py`): Hydra-configured trainer with mixed precision support.
- **Export tooling** (`export/`): emits ONNX graphs, tinygrad-compatible weight pickles, and
  metadata matching the runtime parser expectations.
- **Windows tooling** (`windows/`): PowerShell scripts to bootstrap Python, CARLA, and provide
  helper commands for daily workflows.

## Automated pipeline

After running `setup_env.ps1` and activating the virtual environment, the Windows automation
script can execute the entire workflow:

```powershell
cd tools\sunnypilot_training\windows
Import-Module .\env.psm1
Enter-TrainingEnv
# Collect 50/10 episodes, train with the default Hydra config, and export to .\artifacts
.\run_full_training.ps1 `
  -Scenario urban_intersection `
  -TrainEpisodes 50 `
  -ValEpisodes 10 `
  -OutputDir ..\artifacts
```

`run_full_training.ps1` launches CARLA off-screen, verifies connectivity, records the requested
episodes, trains the diffusion vision and policy models, and exports ONNX/tinygrad/metadata blobs.
The script surfaces failures with actionable error messages (for example when CARLA fails to
start or the dataset is incomplete) and prints the absolute paths to all exported artifacts on
success. Useful switches include:

- `-ExportDir <path>` — write ONNX/tinygrad/metadata files to a custom directory.
- `-CheckpointPath <file>` — export an existing checkpoint without retraining.
- `-SkipDataCollection`, `-SkipTraining`, or `-SkipExport` — reuse previously completed steps.
- `-InstallModels` — copy the exported artifacts into `selfdrive/modeld/models` via
  `integration/replace_models.ps1`.
- `-KeepCarlaRunning` — leave the CARLA process alive after data collection for manual work.

## Quick start

The manual commands below remain available when you need per-step control:

1. **Bootstrap the environment**
   ```powershell
   cd tools\sunnypilot_training\windows
   .\setup_env.ps1
   Import-Module .\env.psm1
   Enter-TrainingEnv
   ```
   The script installs Python 3.11.9, recreates the repository virtual environment, and
   resolves every dependency needed for Windows 10 and 11 hosts. When `-InstallCUDA` is
   specified it queries `nvidia-smi` to verify the detected GPU (including the RTX 5060's
   SM 9.0 capability) and selects the matching PyTorch build (CUDA 12.4 for SM ≥ 9, CUDA 12.1
   otherwise).

2. **Launch CARLA**
   ```powershell
   Invoke-Carla -Port 2000
   ```

3. **Collect training data**
   ```powershell
   Invoke-Collector -Scenario urban_intersection -Output data\zarr\train -Episodes 50
   ```

4. **Train the models**
   ```powershell
   Invoke-Trainer -Config default -OutputDir artifacts
   ```

5. **Export checkpoints**
   ```powershell
   python ..\export\export_vision.py artifacts\checkpoints\epoch_0020.pt --onnx vision.onnx
   python ..\export\export_policy.py artifacts\checkpoints\epoch_0020.pt --onnx policy.onnx
   ```

6. **Install into sunnypilot**
   ```powershell
   ..\integration\replace_models.ps1 -VisionOnnx vision.onnx -VisionTinygrad driving_vision_tinygrad.pkl `
     -VisionMetadata driving_vision_metadata.pkl -PolicyOnnx policy.onnx -PolicyTinygrad driving_policy_tinygrad.pkl `
     -PolicyMetadata driving_policy_metadata.pkl
   ```

## Configuration

Training is driven by Hydra configs stored under `training/configs/`. The default configuration
includes:

- Mixed precision on CUDA (with CPU fallback)
- Cosine LR schedule with warmup
- Photometric jitter + temporal dropout augmentations
- Periodic checkpointing to the working directory

Override parameters on the command line, for example to change the dataset path:

```powershell
python training\train.py dataset.train_path=D:\\dataset\\train dataset.val_path=D:\\dataset\\val
```

## Testing

Unit tests are provided under `tests/` and can be executed with:

```bash
pytest tools/sunnypilot_training/tests
```

## Additional resources

- [Step-by-step training guide](./STEP_BY_STEP.md)
- [CARLA documentation](https://carla.org)
- [Hydra configuration framework](https://hydra.cc)
- [tinygrad project](https://github.com/geohot/tinygrad)
