from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import zmq


def build_response(header: dict[str, Any], horizon_s: float) -> dict[str, Any]:
  seq = int(header.get("seq", 0))
  mono_time_ns = int(header.get("mono_time_ns", 0))
  car_state = header.get("car_state") or {}
  v_ego = float(car_state.get("v_ego", 0.0))

  times = np.linspace(0.0, horizon_s, num=10)
  x = v_ego * times
  y = np.zeros_like(times)

  trajectory = [{"t": float(t), "x": float(xi), "y": float(yi)} for t, xi, yi in zip(times, x, y)]

  return {
    "seq": seq,
    "mono_time_ns": mono_time_ns,
    "horizon_s": horizon_s,
    "trajectory": trajectory,
    "confidence": 0.8,
    "velocity": v_ego,
    "status": "dummy",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Reference DiffusionDrive RPC server")
  parser.add_argument("--bind", default="tcp://0.0.0.0:5555", help="Endpoint to bind the ZeroMQ REP socket")
  parser.add_argument("--horizon", type=float, default=6.0, help="Trajectory horizon in seconds")
  args = parser.parse_args()

  context = zmq.Context.instance()
  socket = context.socket(zmq.REP)
  socket.bind(args.bind)
  print(f"DiffusionDrive dummy server listening on {args.bind}")

  try:
    while True:
      frames = socket.recv_multipart()
      if not frames:
        continue
      header = json.loads(frames[0].decode("utf-8"))
      start = time.perf_counter()
      response = build_response(header, args.horizon)
      response["model_execution_time"] = time.perf_counter() - start
      socket.send_json(response)
  except KeyboardInterrupt:
    print("Stopping DiffusionDrive dummy server")
  finally:
    socket.close(linger=0)
    context.term()


if __name__ == "__main__":
  main()

