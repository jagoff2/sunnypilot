# DIY ASM2464 dock

This checkout defaults to `CHESTNUT_DOCK=asm2464` in `launch_env.sh` for the
previously flashed DIY USB-to-PCIe dock. The observed device is `3801:0001`,
manufacturer `tiny`, product `custom 6e1e151e-CLEAN`. It uses the same USB
identity as Chestnut, so USB identity alone cannot select the supply-sensor policy.

Both model runners start the small model immediately. An isolated GPU worker
probes the bridge before loading an eGPU model.
They require matching bundled firmware and a ready PCIe link (`LTSSM=0x78`).
If PCIe is off, the probe requests PCIe power using the firmware's existing
control command. No flash, USB reset, or driver detach is performed.
Failed probes time out and retain the small-model fallback.

In `asm2464` mode, the dock's successful all-zero supply response means the INA
sensor is unavailable. It is accepted only with a ready PCIe link. Supply faults,
nonzero readings below 5 V, failed transfers, and short responses still block
eGPU startup. Status reporting keeps missing telemetry distinct from power loss;
a lost PCIe link still raises the PCIe alert.

Set `CHESTNUT_DOCK=chestnut` before launching to retain the production dock's
5 V supply requirement. An unset variable outside the launch script, or an
unrecognized value, also uses this stricter policy.

The `chestnutd` service keeps watching for the dock after startup, including when
Quick Boot skips the normal build. A single dock with matching firmware must
remain connected at 5,000 Mbps or faster for three seconds. While offroad, the
service builds a missing stock big model automatically using the ONNX already
in this checkout. It also accepts a chunked ONNX. The latest build output is in
`/tmp/chestnut-build.log`.

Failed or skipped builds retry after 30 seconds, increasing to a maximum of
five minutes between attempts. A new USB device number resets this delay. An
existing manual/model build is allowed to finish first. A disconnect, a new
USB device number, going onroad, or a one-hour timeout stops the complete build
process group. Compilation retries while offroad. `SKIP_TINYGRAD_COMPILE`
continues to disable automatic compilation.

Detection and inference recovery continue onroad, including while moving or
engaged. The small model runs on every frame, keeping its recurrent history
current while the isolated GPU process loads, warms up, or reconnects. Recovery
never restarts the model publisher or controls. The GPU process uses normal
scheduling on CPUs 4/5 during loading, away from the model publisher on CPU 7,
and is preferred for termination under memory pressure. Once loaded, inference
uses CPU 7 at priority 53, below the model publisher's priority 54.

Camera images pass through shared memory. GPU output is accepted only for the
same frame being published, within a 45 ms total inference budget. Late results
are discarded and the current small-model result is used instead. A recovered
GPU must first fill its live recurrent history and produce three consecutive
timely results. A stalled inference process is discarded after 500 ms; startup
has a separate 180-second timeout. Failed attempts back off from 30 seconds to
five minutes. A changed USB device number starts a fresh three-second settle
period. This releases stale GPU handles without stopping small-model output.

The worker uses the current runner's stock or custom eGPU model. Changing
between the stock and custom runner remains a model-selection/startup action.
First-time model compilation remains offroad; an already compiled model can
recover onroad without an ignition cycle. The GPU icon and connection alerts
recover after successful inference. Worker diagnostics are saved to
`/tmp/chestnut-inference.log`.

Driving alerts use fresh, valid, correctly paced `modelV2` output instead of
per-frame `ChestnutActive`/`ChestnutLoading` transitions. Background loading or
switching to a healthy small-model result does not block engagement or emit
`Big Model Failed`. `Big Model Ready` is announced once per drive, after two
seconds of continuously healthy big-model output. Loss of usable model output
still blocks engagement and retains failure protection. Normal communication
and localization checks continue throughout background GPU recovery.

In DIY mode, `chestnutd` keeps the external SuperSpeed root hub at
`power/control=on` and its `a600000.ssusb` parent at `power/control=auto`,
including onroad. The active root hub keeps the parent awake while attached.
After detach, the parent must be allowed to suspend so the next runtime resume
reinitializes the QMP PHY, which the kernel powers down on disconnect.
Forcing the parent to stay awake can prevent that reinitialization. The service
wakes the root hub before restoring the parent to `auto`, and rechecks every
five seconds to handle root-hub recreation. Paths are checked against the
external controller so the internal modem controller is excluded. Production
Chestnut mode retains its existing power-management policy.

The watcher does not reflash firmware or reset the USB controller. It can retry
devices that enumerate late or reconnect, but a dock that never enumerates
still needs its underlying USB/power problem resolved.

Device verification on 2026-09-06 used the stock compiled big model on a comma
four with ignition on, stationary and disengaged. The earlier policy forced
both the external controller and root hub awake, restoring the attached dock
at 5,000 Mbps in that test. That watcher also restored `power/control=on` within
4.2 seconds after it was changed to `auto`. This preceded the correction above
to allow parent suspend and PHY reinitialization on reconnect.
Terminating only the GPU worker caused small-model fallback, followed by an
automatic big-model handoff about 100 seconds later, including backoff, load,
and live-history warmup. The model publisher retained the same PID throughout.
The recovered pipeline averaged about 43 ms and continued publishing at 20 Hz;
occasional late GPU frames used the current small-model output. This was a
stationary hardware test, not a moving or engaged road test. The focused
readiness, hotplug, IPC, and inference recovery suites passed 76 tests.

A subsequent comma reboot exposed a separate unresolved enumeration failure:
the watcher was running and the external controller/root hub were already awake,
but the dock was absent from USB and xHCI repeatedly reported `Cannot set link
state` / `cannot disable (err = -32)`. Allowing runtime suspend, cycling the
external host role, and requesting a USB-C detach/reattach did not recover it.
Those unsuccessful role changes are not automated. The working inference retry
path cannot recover a dock until USB enumeration succeeds; a physical cable
reconnect, connector flip, and power cycles did not consistently restore it.
A later reboot enumerated the same flashed dock at only 12 Mbps on USB2 while
USB3 remained disconnected. At that speed, read-only control requests still
returned a healthy GPU PCIe link (`0x78`) and the expected DIY supply response.
The current blocker is the USB SuperSpeed link, with the specific cause still
unresolved. Runtime inference is deliberately gated on at least 5,000 Mbps.
A single targeted USB reset of the enumerated dock changed its USB address
from 6 to 11, but it returned at the same 12 Mbps. The C4 and its driving
processes were not restarted. This unsuccessful reset is not automated.

The alert integration correction passed 94 focused tests, plus a native comma
four check using the actual selfdrived event and control-state classes. Healthy
small-model fallback kept the simulated control state enabled; lost model output
still blocked engagement and entered soft disable. Deploying the source files
does not reload an already running selfdrived process.
