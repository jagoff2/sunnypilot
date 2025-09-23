# Step-by-step training guide

This walkthrough assumes a Windows 10/11 workstation with a recent NVIDIA driver and
PowerShell 5.1 or later. The same steps work on bare metal or inside a GPU-enabled
virtual machine.

## 0. Automate the entire pipeline (optional)

If you prefer a single command that handles data collection, training, export, and optional
model installation, run the Windows automation script after completing `setup_env.ps1`:

```powershell
cd tools\sunnypilot_training\windows
Import-Module .\env.psm1
Enter-TrainingEnv
.\run_full_training.ps1 -Scenario urban_intersection -TrainEpisodes 50 -ValEpisodes 10 -OutputDir ..\artifacts
```

The script launches CARLA in off-screen mode, waits for a healthy connection, records the
requested episodes into `tools\sunnypilot_training\data\zarr`, trains using the selected Hydra
configuration, and exports ONNX/tinygrad/metadata artifacts into the specified output directory.
Key parameters include:

- `-ExportDir <path>` — choose where the final ONNX/tinygrad/metadata files are written.
- `-CheckpointPath <file>` — export from an existing checkpoint instead of retraining.
- `-SkipDataCollection`, `-SkipTraining`, `-SkipExport` — reuse existing artifacts when iterating.
- `-InstallModels` — copy the exports into `selfdrive/modeld/models` via
  `integration/replace_models.ps1`.
- `-KeepCarlaRunning` — prevent the script from shutting down CARLA after data collection.

The remaining sections describe the manual steps if you want granular control or need to
understand each component in detail.

## 1. Prepare the environment

1. **Open PowerShell as Administrator** (needed the first time to install dependencies).
2. **Navigate to the project**
   ```powershell
   cd path\to\sunnypilot\tools\sunnypilot_training\windows
   ```
3. **Bootstrap Python, CARLA, and the virtual environment**
   ```powershell
   .\setup_env.ps1 [-InstallCUDA]
   Import-Module .\env.psm1
   Enter-TrainingEnv
   ```
   The script installs Python 3.11.9, refreshes the repository `venv/` directory, pulls the
   CARLA 0.10.0 Python API, and installs the full training dependency set. With `-InstallCUDA`
   it inspects `nvidia-smi` to confirm the active GPU (logging the RTX 5060's SM 9.0 reading)
   and automatically chooses the correct PyTorch wheel (CUDA 12.4 for SM ≥ 9, CUDA 12.1 for
   earlier architectures). It also sets the `CARLA_ROOT` user environment variable for later
   commands.
4. **Verify GPU visibility**
   ```powershell
   python - <<'PY'
   import torch
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("Device:", torch.cuda.get_device_name(0))
   PY
   ```

## 2. Launch CARLA with windshield-aligned cameras

1. In a second PowerShell window (same directory), start CARLA headless:
   ```powershell
   Import-Module .\env.psm1
   Invoke-Carla -Port 2000
   ```
2. CARLA uses the `CARLA_ROOT` path configured by `setup_env.ps1`. Leave this window running while collecting data.

## 3. Collect synthetic driving data

1. Back in the training shell, choose an output directory for shards:
   ```powershell
   $trainDir = "${PWD}\..\..\data\zarr\train"
   $valDir = "${PWD}\..\..\data\zarr\val"
   New-Item $trainDir -ItemType Directory -Force | Out-Null
   New-Item $valDir -ItemType Directory -Force | Out-Null
   ```
2. Run the collector for multiple scenarios. The example below gathers 50 training and
   10 validation episodes using the `urban_intersection` registry entry.
   ```powershell
   Invoke-Collector -Scenario urban_intersection -Episodes 50 -Output $trainDir
   Invoke-Collector -Scenario urban_intersection -Episodes 10 -Output $valDir
   ```
   Each episode writes a `.zarr` shard containing:
   - `img` / `big_img`: 12×128×256 Y-channel stacks aligned with the narrow/wide cameras.
   - Vision targets such as `lane_lines` (shape 4×33×2×2) and `lead` (3×6×4×2).
   - Policy inputs (`features_buffer`, `desire`, `traffic_convention`) and outputs (`plan`, `desire_state`).
3. **Sanity check a shard**
   ```powershell
   python - <<'PY'
   import glob, zarr
   shard = glob.glob(r"$trainDir\*.zarr")[0]
   store = zarr.open_group(shard, mode='r')
   print("Inspecting:", shard)
   print("Keys:", list(store.array_keys()))
   print("img shape:", store['img'].shape)
   print("plan shape:", store['plan'].shape, "(IDX_N, PLAN_WIDTH, 2)")
   PY
   ```

## 4. Configure training

1. Copy the default Hydra config if you want to customize it:
   ```powershell
   Copy-Item ..\training\configs\default.yaml ..\training\configs\my_run.yaml
   ```
2. Edit `my_run.yaml` to point at the collected shards, adjust batch size, or tweak diffusion steps.
   At minimum set:
   ```yaml
   dataset:
     train_path: C:/path/to/data/zarr/train
     val_path: C:/path/to/data/zarr/val
   ```

## 5. Launch the trainer

1. From `tools\sunnypilot_training`, run:
   ```powershell
   python training\train.py --config-name my_run
   ```
2. The trainer prints epoch summaries and writes checkpoints to the current working
   directory (default `./artifacts`). Verify GPU utilisation with `nvidia-smi` in another shell.
3. To resume a run, pass `training.resume=true checkpoint=path\to\epoch_xxxx.pt` in Hydra overrides.

## 6. Evaluate the dataset and models (optional but recommended)

1. **Run unit tests** inside the virtual environment:
   ```powershell
   pytest tools\sunnypilot_training\tests
   ```
2. **Inspect a checkpoint**
   ```powershell
   python - <<'PY'
   import torch
   ckpt = torch.load(r"artifacts\checkpoints\epoch_0005.pt", map_location='cpu')
   print("Vision keys:", ckpt['vision'].keys())
   print("Policy keys:", ckpt['policy'].keys())
   PY
   ```

## 7. Export ONNX, tinygrad weights, and metadata

1. Export the vision model:
   ```powershell
   python export\export_vision.py artifacts\checkpoints\epoch_0020.pt `
     --onnx artifacts\vision.onnx `
     --tinygrad artifacts\driving_vision_tinygrad.pkl `
     --metadata artifacts\driving_vision_metadata.pkl
   ```
2. Export the policy model:
   ```powershell
   python export\export_policy.py artifacts\checkpoints\epoch_0020.pt `
     --onnx artifacts\policy.onnx `
     --tinygrad artifacts\driving_policy_tinygrad.pkl `
     --metadata artifacts\driving_policy_metadata.pkl
   ```
3. Validate the exported metadata matches sunnypilot contracts:
   ```powershell
   python - <<'PY'
   import pickle
   meta = pickle.load(open('artifacts\\driving_vision_metadata.pkl', 'rb'))
   print(meta['output_slices'])
   meta = pickle.load(open('artifacts\\driving_policy_metadata.pkl', 'rb'))
   print(meta['input_shapes'])
   PY
   ```

## 8. Replace models in a sunnypilot install

1. Stop any running sunnypilot instance.
2. Run the integration helper from Windows PowerShell:
   ```powershell
   tools\sunnypilot_training\integration\replace_models.ps1 `
     -VisionTinygrad artifacts\driving_vision_tinygrad.pkl `
     -VisionMetadata artifacts\driving_vision_metadata.pkl `
     -PolicyTinygrad artifacts\driving_policy_tinygrad.pkl `
     -PolicyMetadata artifacts\driving_policy_metadata.pkl
   ```
   The script copies artifacts into `%LOCALAPPDATA%\comma\modeld` and triggers a smoke test using the
   shipped parser to verify shapes and slice alignment.

## 9. Post-run housekeeping

- Archive checkpoints together with the Hydra config to reproduce the run later.
- Keep CARLA shards versioned (map, weather seed, scenario registry commit) for auditability.
- Periodically run `pytest` to ensure schema or metadata changes remain compatible.

Following this guide produces contract-compliant diffusion models that plug into
`sunnypilot/selfdrive/modeld` without manual tensor surgery.
