from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from msgq.visionipc import VisionBuf, VisionIpcClient, VisionStreamType
from openpilot.common.swaglog import cloudlog


@dataclass
class FrameMetadata:
  """Metadata associated with a VisionIPC frame."""
  frame_id: int
  timestamp_sof: int
  timestamp_eof: int

  @classmethod
  def from_client(cls, client: VisionIpcClient) -> "FrameMetadata":
    return cls(frame_id=client.frame_id, timestamp_sof=client.timestamp_sof, timestamp_eof=client.timestamp_eof)


def _format_from_stream(stream_type: VisionStreamType) -> str:
  if stream_type == VisionStreamType.VISION_STREAM_ROAD:
    return "road_wide_nv12"
  if stream_type == VisionStreamType.VISION_STREAM_DRIVER:
    return "driver_nv12"
  if stream_type == VisionStreamType.VISION_STREAM_MAP:
    return "map_nv12"
  return "wide_nv12"


class VisionStreamReader:
  """Helper for connecting to a VisionIPC stream and extracting frames."""

  def __init__(self, stream_type: VisionStreamType, client_name: str = "camerad", conflate: bool = True):
    self.stream_type = stream_type
    self.client = VisionIpcClient(client_name, stream_type, conflate)
    self._format = _format_from_stream(stream_type)

  @property
  def format(self) -> str:
    return self._format

  def connect(self, block: bool = True, retry_delay: float = 0.1) -> None:
    if self.client.is_connected():
      return
    while not self.client.connect(block):
      if not block:
        break
      cloudlog.debug("vision stream connect retry %s", self.stream_type)
      time.sleep(retry_delay)

  def recv(self, timeout_ms: int = 100) -> tuple[Optional[VisionBuf], Optional[FrameMetadata]]:
    buf = self.client.recv(timeout_ms)
    if buf is None:
      return None, None
    return buf, FrameMetadata.from_client(self.client)

  @property
  def width(self) -> int:
    return self.client.width or 0

  @property
  def height(self) -> int:
    return self.client.height or 0

  @property
  def stride(self) -> int:
    return self.client.stride or 0

  def frame_bytes(self, buf: VisionBuf, copy: bool = False) -> bytes:
    data = buf.data
    if copy:
      return bytes(data)
    return memoryview(data)

  def yuv_image(self, buf: VisionBuf) -> np.ndarray:
    return buf.data.reshape((-1, self.stride))

