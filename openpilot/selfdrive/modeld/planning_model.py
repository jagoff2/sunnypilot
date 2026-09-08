"""Select a complete, bounded-age model for downstream planning."""
import copy
import math
import time

from openpilot.selfdrive.modeld.lane_centering_safety import LaneCenteringSafetyLatch


class PlanningModelCache:
  def __init__(self):
    self.guard = LaneCenteringSafetyLatch()
    self.source_mono_time = 0

  @staticmethod
  def complete_trajectory(model):
    # These vectors are consumed together by MPC, SCC, and DEC. An invalid
    # producer packet intentionally leaves them empty; never pass it through.
    return all(len(values := getattr(getattr(model, field), axis)) == 33 and all(map(math.isfinite, values))
               for field in ('position', 'velocity', 'acceleration', 'orientation', 'orientationRate')
               for axis in ('x', 'y', 'z'))

  def update(self, sm, now_ns):
    model = sm['modelV2']
    healthy = (sm.seen['modelV2'] and sm.all_alive(['modelV2']) and sm.all_valid(['modelV2']) and
               self.complete_trajectory(model))
    previous = self.guard.last_usable_model
    # Only the consumer's selection is used here. The engagement latch remains
    # owned by controlsd/selfdrived and cannot be cleared by this planner.
    self.guard.update(model, healthy, sm.updated['modelV2'], now_ns, engagement_requested=False)
    selected = self.guard.selected_model
    if selected is None:
      return None
    new_model = selected is not previous
    if new_model:
      self.source_mono_time = sm.logMonoTime['modelV2']

    # Keep every non-model input current, including its original health. Never
    # mutate SubMaster's received packet, age, or watchdog bookkeeping.
    view = copy.copy(sm)
    for field in ('data', 'logMonoTime', 'alive', 'valid', 'freq_ok', 'updated'):
      setattr(view, field, getattr(sm, field).copy())
    view.data['modelV2'] = selected
    view.logMonoTime['modelV2'] = self.source_mono_time
    view.updated['modelV2'] = new_model
    # This health belongs to the selected complete model within the command-age
    # limit, including during a rejected packet or a model frequency dip.
    view.alive['modelV2'] = view.valid['modelV2'] = view.freq_ok['modelV2'] = True
    return view


class PlannerCadence:
  """Run at the planner's fixed step without bursts after a delayed tick."""
  def __init__(self, period, clock=time.monotonic, sleep=time.sleep):
    self.period, self.clock, self.sleep = period, clock, sleep
    self.next_frame = clock()

  def wait(self):
    now = self.clock()
    if now < self.next_frame:
      self.sleep(self.next_frame - now)
    self.next_frame = max(self.next_frame, self.clock()) + self.period
