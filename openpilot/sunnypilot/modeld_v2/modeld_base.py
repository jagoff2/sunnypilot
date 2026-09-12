"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.sunnypilot.livedelay.helpers import DEFAULT_LAT_DELAY, get_lat_delay


class ModelStateBase:
  def __init__(self, params=None, initial_lat_delay=DEFAULT_LAT_DELAY):
    self.lat_delay = get_lat_delay(params if params is not None else Params(), initial_lat_delay, initial_lat_delay)
