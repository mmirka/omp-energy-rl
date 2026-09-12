#!/usr/bin/env bash
#
# launch_control.sh -- start a workload and the online controller alongside it.
#
# The two processes are pinned to disjoint sets of cores:
#
#   workload   -> 0x7FFFF = cores 0..18   (the controller then narrows this)
#   controller -> 0x80000 = core 19       (pinned at 3.10 GHz)
#
# Keeping them disjoint is not tidiness. The controller runs a Keras training
# step every 500 ms; if it shared cores with the workload it would perturb the
# very CpS and CpJ it is trying to measure, and the reward signal would partly
# reflect the controller's own load. Core 19 is held at a high frequency for the
# same reason -- the measurement must never become the bottleneck. That is also
# why the machine has 20 cores but the action space only reaches 19.
#
#   usage: sudo ./launch_control.sh <program> [args...]
#   e.g.:  sudo ./launch_control.sh ../bench/two_phase 1000000000 5000000 20000000 19
#
# Environment:
#   ENCODER, SCALER  -- enable phase-aware control (thesis Fig. 4.17)
#   PRESET           -- agent preset; see omp_energy_rl.agent.PRESETS
#   STEPS            -- sampling periods to run (default 3000 = 25 minutes)
#
# Requires the testbed and a workload built against the patched GCC/libgomp.

set -euo pipefail

WORKLOAD_MASK=0x7FFFF
CONTROLLER_MASK=0x80000
STEPS=${STEPS:-3000}
PRESET=${PRESET:-thesis-table-4.3}

[ $# -ge 1 ] || { echo "usage: $0 <program> [args...]" >&2; exit 1; }
PROGRAM=$1; shift

taskset "$WORKLOAD_MASK" "$PROGRAM" "$@" &
WORKLOAD_PID=$!
trap 'kill "$WORKLOAD_PID" 2>/dev/null || true' EXIT

# Give the workload time to allocate and first-touch its working set. Starting
# the controller earlier would spend the first slice of the exploration period
# rewarding actions against a startup transient rather than the steady workload.
sleep 5

ARGS=(--pid "$WORKLOAD_PID" --preset "$PRESET" --steps "$STEPS")
if [ -n "${ENCODER:-}" ]; then
    ARGS+=(--encoder "$ENCODER" --scaler "${SCALER:?SCALER must be set alongside ENCODER}")
fi

taskset "$CONTROLLER_MASK" python3 ../run_controller.py "${ARGS[@]}"
