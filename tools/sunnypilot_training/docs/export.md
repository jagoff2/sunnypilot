# Export workflow

The export scripts transform trained checkpoints into artifacts consumable by sunnypilot's
`modeld` runtime.

## Vision model

```powershell
python tools\sunnypilot_training\export\export_vision.py artifacts\checkpoints\epoch_0020.pt `
  --onnx artifacts\driving_vision.onnx `
  --tinygrad artifacts\driving_vision_tinygrad.pkl `
  --metadata artifacts\driving_vision_metadata.pkl
```

This performs the following steps:

1. Load the checkpoint and instantiate `DiffusionVisionModel`.
2. Export a flattened ONNX graph for interoperability.
3. Serialize the PyTorch state dict into a tinygrad-compatible pickle for runtime execution.
4. Generate metadata containing input shapes and output slices (consumed by `Parser`).

## Policy model

```powershell
python tools\sunnypilot_training\export\export_policy.py artifacts\checkpoints\epoch_0020.pt `
  --onnx artifacts\driving_policy.onnx `
  --tinygrad artifacts\driving_policy_tinygrad.pkl `
  --metadata artifacts\driving_policy_metadata.pkl
```

The policy exporter wraps the diffusion sampler to flatten outputs into a single ONNX tensor and
saves metadata describing the plan mixture layout.

## Installing into sunnypilot

After generating artifacts, run the integration helper to update the runtime models:

```powershell
powershell tools\sunnypilot_training\integration\replace_models.ps1 `
  -VisionOnnx artifacts\driving_vision.onnx `
  -VisionTinygrad artifacts\driving_vision_tinygrad.pkl `
  -VisionMetadata artifacts\driving_vision_metadata.pkl `
  -PolicyOnnx artifacts\driving_policy.onnx `
  -PolicyTinygrad artifacts\driving_policy_tinygrad.pkl `
  -PolicyMetadata artifacts\driving_policy_metadata.pkl
```
