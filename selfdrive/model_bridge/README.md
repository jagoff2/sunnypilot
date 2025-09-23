# DiffusionDrive bridge overview

The DiffusionDrive bridge (`selfdrive.model_bridge.model_bridge`) replaces
`modeld` when the `EnableDiffusionDrive` param is set. It subscribes to the
selected VisionIPC camera stream, forwards calibrated frames to an external
policy server, and republishes the returned trajectory as a full
`modelV2`/`drivingModelData` bundle so that the remainder of the pipeline stays
untouched.

## Runtime data flow

1. Frames are read from VisionIPC (`road`, `wide`, `driver`, or `map`).
2. Calibration is refreshed whenever `liveCalibration` publishes new roll/pitch/
yaw. The warp matrix, intrinsics, and camera metadata are included in the
header that is sent to the host.
3. Each frame generates a two-part ZeroMQ REQ payload: a JSON header with
calibration, car/device state, frame timing, and an opaque `seq` counter, and a
second part containing the raw NV12 buffer.
4. The external server is expected to respond with a JSON document containing
the original `seq`, monotonic timestamp, and a future ego trajectory expressed
in the device frame (see [tools/diffusiondrive/README.md](../../tools/diffusiondrive/README.md)).
5. The adapter resamples the trajectory to `ModelConstants.T_IDXS`, computes
velocities, longitudinal acceleration, yaw, yaw rate, curvature, and generates
the `modelV2`, `drivingModelData`, `modelDataV2SP`, and `cameraOdometry`
messages.
6. The bridge writes `DiffusionDriveStatus` for UI/status inspection and
republishes the model events on the existing sockets so planner/control remain
stock.

## Parameters

| Param | Purpose |
| --- | --- |
| `EnableDiffusionDrive` | Toggles the bridge on (disabling stock `modeld`). |
| `DiffusionDriveEndpoint` | Persisted ZeroMQ endpoint used by the bridge. |
| `DiffusionDriveStatus` | JSON health snapshot including connectivity, seq, and latency. |

Parameters are declared in `common/params_keys.h`. The bridge writes
`DiffusionDriveEndpoint` when started, even if the param did not previously
exist, and keeps `DiffusionDriveStatus` updated with connection health, current
sequence, and latency. When timeouts or transport errors occur, the status flag
is updated to reflect the failure so off-road diagnostics and UI plumbing can
react.

## Model message mapping

The adapter (`adapter.py`) performs the following conversions:

* Trajectory resampling uses linear interpolation at `ModelConstants.T_IDXS` so
  the resulting arrays line up with planner expectations.
* Velocity, acceleration, and yaw rate are derived directly from the resampled
  path. Longitudinal acceleration is approximated by the derivative of speed,
  with an additional correction nudging toward the supplied velocity reference
  if the trajectory is perfectly constant.
* Curvature uses the standard planar definition
  \(\kappa = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}}\). The initial value is
  exposed to both `model.action.desiredCurvature` and
  `drivingModelData.action.desiredCurvature`.
* `shouldStop` honours `meta.should_stop`, `meta.shouldStop`, or `meta.stop` if
  present, otherwise defaults to `False`.
* Lane lines, road edges, and leads are populated with zero-probability stubs
  so controls can continue to run while DiffusionDrive provides the lateral
  trajectory. These values can be replaced by richer estimates later without
  touching the bridge.
* `model.meta.engagedProb` defaults to the reported confidence when the host
  does not supply an explicit value.

All generated Cap’n Proto messages set the `valid` bit based on both the
adapter accepting the host trajectory and calibration being available.

## Transport contract

The bridge currently uses a ZeroMQ REQ socket with a configurable timeout
(default 0.5 s). Payload structure:

* **header (JSON):**
  ```json
  {
    "seq": 123,
    "mono_time_ns": 171234567890,
    "frame_id": 4567,
    "width": 1928,
    "height": 1208,
    "stride": 2048,
    "format": "wide_nv12",
    "calibration": {"warp_matrix": [[...]], ...},
    "car_state": {"v_ego": 12.3, "a_ego": 0.1, "steering_angle_deg": -0.5},
    "frame_drop_ratio": 0.0
  }
  ```
* **frame bytes:** NV12 buffer covering the VisionIPC frame.

Replies must include `seq`, `mono_time_ns`, `horizon_s`, `confidence`, and a
`trajectory` array of `{t, x, y, z}` waypoints sorted by time. Optional fields
such as `velocity`, `meta`, and `model_execution_time` are forwarded where
available.

## Failure handling

* Timeouts, transport errors, and malformed responses are logged, counted, and
  reflected in `DiffusionDriveStatus`. The bridge keeps running and will retry
  on the next frame.
* If calibration has not been seen yet the messages are marked invalid, keeping
  planners cautious until a valid transform is available.
* Disabling `EnableDiffusionDrive` immediately reverts to stock `modeld`
  without a restart.

## Bench testing

1. On the host PC, run the dummy or real policy server (see
   [tools/diffusiondrive/README.md](../../tools/diffusiondrive/README.md)).
2. On-device, set `EnableDiffusionDrive` to `1` using `./tools/one_shot_param_setter` or `op_params`.
3. Start the bridge manually for bench testing:
   ```bash
   python3 -m selfdrive.model_bridge.model_bridge --endpoint tcp://PC_IP:5555
   ```
4. Inspect `/tmp/params/d/DiffusionDriveStatus` or the on-device UI for
   latency and connection status updates.

Once confidence is established with recorded logs and the dummy server, connect
the Windows workstation running the full DiffusionDrive model to exercise the
complete stack.
