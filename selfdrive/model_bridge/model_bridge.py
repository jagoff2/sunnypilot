from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import cereal.messaging as messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants

from msgq.visionipc import VisionStreamType

from .adapter import BridgeOutputs, DiffusionDriveAdapter, DiffusionDriveResponse
from .calib import CalibrationManager
from .ipc import FrameMetadata, VisionStreamReader
from .transport import (DiffusionDriveClient, DiffusionDriveResponseError,
                        DiffusionDriveStatus, DiffusionDriveTimeout,
                        DiffusionDriveTransportError)


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
STATUS_PARAM = "DiffusionDriveStatus"
ENDPOINT_PARAM = "DiffusionDriveEndpoint"


@dataclass
class BridgeConfig:
  endpoint: str
  timeout_s: float
  stream: VisionStreamType


def _safe_get(params: Params, key: str) -> str | None:
  try:
    return params.get(key)
  except UnknownKeyName:
    return None


def _resolve_stream(name: str | None) -> VisionStreamType:
  if not name:
    return VisionStreamType.VISION_STREAM_WIDE_ROAD
  lower = name.lower()
  if lower in ("road", "fcam"):
    return VisionStreamType.VISION_STREAM_ROAD
  if lower in ("driver", "dcam"):
    return VisionStreamType.VISION_STREAM_DRIVER
  if lower in ("map", "nav"):
    return VisionStreamType.VISION_STREAM_MAP
  return VisionStreamType.VISION_STREAM_WIDE_ROAD


def load_config(params: Params, args: argparse.Namespace) -> BridgeConfig:
  endpoint = args.endpoint or os.getenv("DIFFUSIONDRIVE_ENDPOINT") or _safe_get(params, ENDPOINT_PARAM) or DEFAULT_ENDPOINT
  timeout_s = args.timeout or float(os.getenv("DIFFUSIONDRIVE_TIMEOUT", "0.5"))
  stream_name = args.stream or os.getenv("DIFFUSIONDRIVE_STREAM")
  stream = _resolve_stream(stream_name)
  return BridgeConfig(endpoint=endpoint, timeout_s=timeout_s, stream=stream)


def write_status(params: Params, status: DiffusionDriveStatus) -> None:
  payload = {
    "endpoint": status.endpoint,
    "connected": status.connected,
  }
  if status.latency_ms is not None:
    payload["latency_ms"] = status.latency_ms
  if status.seq is not None:
    payload["seq"] = status.seq
  if status.detail:
    payload["detail"] = status.detail
  try:
    params.put_nonblocking(STATUS_PARAM, json.dumps(payload))
  except UnknownKeyName:
    cloudlog.debug("status param %s not registered", STATUS_PARAM)


def build_header(seq: int, frame: FrameMetadata, reader: VisionStreamReader,
                 calibration: CalibrationManager, sm: messaging.SubMaster,
                 frame_drop_ratio: float) -> dict:
  car_state = sm['carState'] if sm.seen['carState'] else None
  device_state = sm['deviceState'] if sm.seen['deviceState'] else None
  road_camera_state = sm['roadCameraState'] if sm.seen['roadCameraState'] else None

  header = {
    "seq": seq,
    "mono_time_ns": frame.timestamp_sof,
    "timestamp_eof": frame.timestamp_eof,
    "frame_id": frame.frame_id,
    "width": reader.width,
    "height": reader.height,
    "stride": reader.stride,
    "format": reader.format,
    "frame_drop_ratio": frame_drop_ratio,
    "calibration": calibration.payload(),
  }

  if car_state is not None:
    header["car_state"] = {
      "v_ego": float(car_state.vEgo),
      "a_ego": float(car_state.aEgo),
      "steering_angle_deg": float(car_state.steeringAngleDeg),
    }
  if device_state is not None:
    header["device_type"] = str(device_state.deviceType)
  if road_camera_state is not None:
    header["road_camera_state"] = {
      "frame_id": int(road_camera_state.frameId),
      "timestamp_sof": int(road_camera_state.timestampSof),
    }
  return header


def main() -> None:
  parser = argparse.ArgumentParser(description="DiffusionDrive bridge process")
  parser.add_argument("--endpoint", help="ZeroMQ endpoint for DiffusionDrive server")
  parser.add_argument("--timeout", type=float, help="Request timeout in seconds")
  parser.add_argument("--stream", choices=["road", "wide", "driver", "map"], help="Vision stream to forward")
  args = parser.parse_args()

  params = Params()
  config = load_config(params, args)
  try:
    params.put_nonblocking(ENDPOINT_PARAM, config.endpoint)
  except UnknownKeyName:
    cloudlog.debug("endpoint param %s not registered", ENDPOINT_PARAM)
  cloudlog.info("DiffusionDrive bridge connecting to %s stream=%s", config.endpoint, config.stream)

  client = DiffusionDriveClient(config.endpoint, request_timeout_s=config.timeout_s)
  reader = VisionStreamReader(config.stream)
  calibration = CalibrationManager(config.stream)
  adapter = DiffusionDriveAdapter()

  frame_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  last_frame_id = 0
  run_count = 0
  seq = 0

  pm = messaging.PubMaster(["modelV2", "drivingModelData", "modelDataV2SP", "cameraOdometry"])
  sm = messaging.SubMaster([
    "deviceState",
    "roadCameraState",
    "carState",
    "liveCalibration",
  ], poll='deviceState')

  write_status(params, DiffusionDriveStatus(endpoint=config.endpoint, connected=False, detail="starting"))

  reader.connect()
  cloudlog.info("Vision stream connected: %s", reader.format)

  try:
    while True:
      buf, frame_meta = reader.recv()
      if buf is None or frame_meta is None:
        continue

      sm.update(0)

      if sm.updated['liveCalibration'] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
        calibration.update(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor), sm['liveCalibration'].rpyCalib)

      live_calib_seen = calibration.data.valid
      planner_frame_id = sm['roadCameraState'].frameId if sm.seen['roadCameraState'] else frame_meta.frame_id

      vipc_dropped_frames = max(0, frame_meta.frame_id - last_frame_id - 1)
      frames_dropped = frame_filter.update(min(vipc_dropped_frames, 10))
      if run_count < 10:
        frame_filter.x = 0.
        frames_dropped = 0.
      run_count += 1
      frame_drop_ratio = frames_dropped / (1 + frames_dropped)

      header = build_header(seq, frame_meta, reader, calibration, sm, frame_drop_ratio)
      payload = reader.frame_bytes(buf)
      seq += 1

      start = time.perf_counter()
      try:
        reply = client.send(header, payload)
      except DiffusionDriveTimeout as err:
        cloudlog.error("DiffusionDrive timeout: %s", err)
        last_frame_id = frame_meta.frame_id
        write_status(params, DiffusionDriveStatus(config.endpoint, connected=False, detail="timeout"))
        continue
      except DiffusionDriveTransportError as err:
        cloudlog.error("DiffusionDrive transport error: %s", err)
        last_frame_id = frame_meta.frame_id
        write_status(params, DiffusionDriveStatus(config.endpoint, connected=False, detail="transport"))
        continue
      except DiffusionDriveResponseError as err:
        cloudlog.error("DiffusionDrive response error: %s", err)
        last_frame_id = frame_meta.frame_id
        write_status(params, DiffusionDriveStatus(config.endpoint, connected=False, detail="response"))
        continue

      total_time = time.perf_counter() - start
      response = DiffusionDriveResponse.from_dict(reply)
      outputs: BridgeOutputs = adapter.build_messages(response, frame_meta, frame_meta,
                                                      planner_frame_id, frame_drop_ratio,
                                                      total_time, live_calib_seen)

      pm.send('modelV2', outputs.model_msg)
      pm.send('drivingModelData', outputs.driving_msg)
      pm.send('modelDataV2SP', outputs.model_sp_msg)
      pm.send('cameraOdometry', outputs.camera_odometry_msg)

      last_frame_id = frame_meta.frame_id
      if not live_calib_seen:
        detail = "calibration_missing"
      elif outputs.valid:
        detail = response.status or "ok"
      else:
        detail = response.status or "invalid"
      write_status(params, DiffusionDriveStatus(
        endpoint=config.endpoint,
        connected=True,
        latency_ms=total_time * 1000.0,
        seq=response.seq,
        detail=detail,
      ))
  except KeyboardInterrupt:
    cloudlog.warning("DiffusionDrive bridge interrupted")
  finally:
    client.close()


if __name__ == "__main__":
  main()

