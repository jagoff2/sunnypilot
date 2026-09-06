# DIY ASM2464 dock

This checkout defaults to `CHESTNUT_DOCK=asm2464` in `launch_env.sh` for the
previously flashed DIY USB-to-PCIe dock. The observed device is `3801:0001`,
manufacturer `tiny`, product `custom 6e1e151e-CLEAN`. It uses the same USB
identity as Chestnut, so USB identity alone cannot select the supply-sensor policy.

Both model runners probe the bridge directly before loading an eGPU model.
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

The stock big model must still be compiled. Connect and power the dock before
running the normal startup build; that build includes the big model when the
matching dock is visible. Hot-plugging a dock into an already running manager
does not start compilation or replace the running small model.
