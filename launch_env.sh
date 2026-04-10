#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

MODEL_PATH="/sys/firmware/devicetree/base/model"
if [ -r "$MODEL_PATH" ]; then
  MODEL="$(tr -d '\0' < "$MODEL_PATH")"
  if [ "$MODEL" = "comma tizi" ]; then
    # Force USB AMD compilation/runtime on C3X boots so rebuilds don't fall back to QCOM.
    export USBGPU=1
  fi
fi

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="17.2"
fi

export STAGING_ROOT="/data/safe_staging"
