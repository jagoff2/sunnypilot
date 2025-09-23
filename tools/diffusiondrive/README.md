# DiffusionDrive host utilities

This directory contains the tooling that pairs with
[`selfdrive/model_bridge`](../../selfdrive/model_bridge/README.md). The bridge on
an Openpilot-compatible device streams calibrated VisionIPC frames to a PC and
expects a policy response in the stock `modelV2` format. The files here help you
bootstrap the Windows workstation that will run DiffusionDrive and provide a
dummy policy for latency testing.

## Host PC prerequisites

* Windows 10/11 with administrator rights.
* NVIDIA RTX 20xx-class GPU or better with up-to-date Studio/Game Ready driver.
* Python 3.10 or newer added to `PATH`.
* Git (with Git LFS enabled) and PowerShell 7+.
* Optional but recommended: Visual Studio 2019/2022 Build Tools (installs the
  MSVC toolchain required by some Python packages).
* Sufficient free disk space (~15 GB) for the DiffusionDrive repo, virtual
  environment, and downloaded weights.

## Quick start (PowerShell)

1. Open **PowerShell** as Administrator.
2. Navigate to the sunnypilot checkout:
   ```powershell
   cd C:\path\to\sunnypilot\tools\diffusiondrive
   ```
3. Execute the installer:
   ```powershell
   .\install_windows.ps1 -InstallDir C:\DiffusionDrive -TorchIndexUrl https://download.pytorch.org/whl/cu118
   ```

   The script performs the following steps:

   * Verifies that `python` and `git` are available.
   * Clones `https://github.com/hustvl/DiffusionDrive` (or updates an existing checkout).
   * Creates a virtual environment under `<InstallDir>\dd-env`.
   * Installs PyTorch/Torchvision/Torchaudio from the supplied CUDA wheel index.
   * Installs the upstream DiffusionDrive requirements and the bridge server
     dependencies (`tools/diffusiondrive/server/requirements.txt`).
   * Downloads the open-weights release from Hugging Face into `<InstallDir>\weights`
     using `download_weights.py`.

   Important flags:

   | Flag | Description |
   | --- | --- |
   | `-InstallDir` | Root folder for the repo, virtual environment, and weights. |
   | `-Python` | Alternate python executable (defaults to `python`). |
   | `-TorchIndexUrl` | Wheel index for PyTorch. Change if you need a different CUDA version. |
   | `-TorchPackages` | Overrides the torch packages list (default `"torch torchvision torchaudio"`). |
   | `-SkipWeights` | Skip the Hugging Face download (useful when weights already exist). |
   | `-HuggingFaceToken` | Personal access token if the model requires authentication. |

   After completion the script prints the commands needed to activate the virtual
   environment and launch the sample server.

## Manual installation

If you prefer to perform the setup yourself, replicate the following steps in a
PowerShell terminal:

1. Clone the upstream repository:
   ```powershell
   git clone https://github.com/hustvl/DiffusionDrive.git C:\DiffusionDrive\DiffusionDrive
   cd C:\DiffusionDrive\DiffusionDrive
   git lfs install
   ```
2. Create a Python environment (replace the path if you use Conda):
   ```powershell
   python -m venv ..\dd-env
   ..\dd-env\Scripts\python.exe -m pip install --upgrade pip
   ```
3. Install PyTorch with the CUDA build matching your GPU driver:
   ```powershell
   ..\dd-env\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Consult the [PyTorch install matrix](https://pytorch.org/get-started/locally/)
   for alternate CUDA versions.
4. Install DiffusionDrive and bridge requirements:
   ```powershell
   ..\dd-env\Scripts\python.exe -m pip install -r requirements.txt
   ..\dd-env\Scripts\python.exe -m pip install -r ..\sunnypilot\tools\diffusiondrive\server\requirements.txt
   ```
5. Download weights with the helper script:
   ```powershell
   ..\dd-env\Scripts\python.exe ..\sunnypilot\tools\diffusiondrive\server\download_weights.py `
     --local-dir ..\weights
   ```
   Add `--token YOUR_HF_TOKEN` if you need to authenticate with Hugging Face.
6. (Optional) Register the weights location so your own inference script can
   discover them:
   ```powershell
   setx DIFFUSIONDRIVE_WEIGHTS "C:\DiffusionDrive\weights"
   ```

## Running the sample server

The included `diffusiondrive_server.py` is a reference ZeroMQ REP server that
replies with a straight-line trajectory. It is invaluable for verifying end-to-end
transport and timing before wiring in the real model. Launch it from the sunnypilot
checkout:

```powershell
C:\DiffusionDrive\dd-env\Scripts\python.exe tools\diffusiondrive\server\diffusiondrive_server.py `
  --bind tcp://0.0.0.0:5555 --horizon 6.0
```

Monitor the bridge status on the device (`DiffusionDriveStatus` param) to confirm
frames are flowing and latency is within budget. Once the dummy loop is stable
replace `build_response` with an adapter that loads the actual DiffusionDrive
policy and returns its predicted trajectory in the documented format.

## Download helper script

`server/download_weights.py` wraps `huggingface_hub.snapshot_download` to grab
only the files relevant to inference (default: `*.ckpt`, `*.pt`, `*.pth`, and
YAML configs). Example usages:

```powershell
# Download the full weights snapshot
python tools\diffusiondrive\server\download_weights.py --local-dir C:\DiffusionDrive\weights

# Restrict to specific files
python tools\diffusiondrive\server\download_weights.py --local-dir C:\DiffusionDrive\weights `
  --pattern "**/diffusiondrive*.ckpt" --pattern "**/config*.yaml"

# Inspect the repository contents without downloading
python tools\diffusiondrive\server\download_weights.py --list-only
```

## Directory layout

```
 tools/diffusiondrive/
 ├── README.md                 ← this document
 ├── install_windows.ps1       ← automated Windows installer
 └── server/
     ├── diffusiondrive_server.py  ← dummy ZeroMQ policy server
     ├── download_weights.py       ← Hugging Face snapshot helper
     └── requirements.txt          ← common host-side Python dependencies
```

Pair these tools with the bridge documentation for an end-to-end walkthrough
covering installation, connectivity checks, and status reporting.
