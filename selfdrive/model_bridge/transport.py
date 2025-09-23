from __future__ import annotations

import json
from dataclasses import dataclass

import zmq


class DiffusionDriveError(Exception):
  """Base exception for bridge transport errors."""


class DiffusionDriveTimeout(DiffusionDriveError):
  pass


class DiffusionDriveTransportError(DiffusionDriveError):
  pass


class DiffusionDriveResponseError(DiffusionDriveError):
  pass


@dataclass
class DiffusionDriveStatus:
  endpoint: str
  connected: bool
  latency_ms: float | None = None
  seq: int | None = None
  detail: str | None = None


class DiffusionDriveClient:
  def __init__(self, endpoint: str, request_timeout_s: float = 0.5, linger_ms: int = 0):
    self.endpoint = endpoint
    self.request_timeout_ms = max(int(request_timeout_s * 1000), 1)
    self.linger_ms = linger_ms
    self._ctx = zmq.Context.instance()
    self._socket: zmq.Socket | None = None
    self._connect()

  def _connect(self) -> None:
    if self._socket is not None:
      self._socket.close(linger=self.linger_ms)
    self._socket = self._ctx.socket(zmq.REQ)
    self._socket.setsockopt(zmq.LINGER, self.linger_ms)
    self._socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
    self._socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
    self._socket.connect(self.endpoint)

  def close(self) -> None:
    if self._socket is not None:
      self._socket.close(linger=self.linger_ms)
      self._socket = None

  def send(self, header: dict, frame_payload: memoryview | bytes | bytearray) -> dict:
    if self._socket is None:
      self._connect()
    assert self._socket is not None

    try:
      header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as err:
      raise DiffusionDriveTransportError(f"failed to encode header: {err}") from err

    try:
      self._socket.send_multipart([header_bytes, frame_payload], copy=False)
    except zmq.ZMQError as err:
      self._connect()
      raise DiffusionDriveTransportError(f"failed to send frame: {err}") from err

    try:
      reply = self._socket.recv()
    except zmq.Again as err:
      self._connect()
      raise DiffusionDriveTimeout("timeout waiting for DiffusionDrive reply") from err
    except zmq.ZMQError as err:
      self._connect()
      raise DiffusionDriveTransportError(f"recv error: {err}") from err

    try:
      return json.loads(reply.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as err:
      raise DiffusionDriveResponseError(f"invalid response payload: {err}") from err

