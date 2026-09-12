#!/usr/bin/env bash
#
# set_config.sh -- apply one (core count, frequency) configuration to the testbed.
#
# This is the shell side of a controller action. The Python controller
# (src/omp_energy_rl/environment.py) issues the same three cpupower calls and the
# same taskset call directly; this script exists so the mechanism can be driven
# by hand, from a sweep, or from a shell experiment without Python in the loop.
#
#   usage: sudo ./set_config.sh <n_cores> <freq_ghz> [workload_pid]
#   e.g.:  sudo ./set_config.sh 8 1.9 $(pidof -s ./mixed_cxmy)
#
# Requires the testbed: 20 cores across 2 sockets, cpupower, and the
# acpi-cpufreq driver (NOT intel_pstate -- see the README's 'The testbed' section).

set -euo pipefail

TOTAL_CORES=20
CORES_PER_SOCKET=10
CONTROLLER_CORE=19          # runs the controller/collector; never given to the workload
CONTROLLER_FREQ=3.10        # pinned high so measurement is never the bottleneck
MIN_FREQ=1.20
MAX_WORKLOAD_CORES=19

usage() {
    echo "usage: $0 <n_cores 1..${MAX_WORKLOAD_CORES}> <freq_ghz 1.2..2.2> [workload_pid]" >&2
    exit 1
}

[ $# -ge 2 ] || usage
N_CORES=$1
FREQ=$2
PID=${3:-}

# Refuse early rather than half-applying a configuration to the wrong machine.
# The core/socket layout is not incidental here: the socket-interleaved masks
# below and the 209-configuration action space are both defined against it.
online_cores=$(getconf _NPROCESSORS_ONLN)
sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
if ! command -v cpupower > /dev/null \
   || [ "$online_cores" != "$TOTAL_CORES" ] \
   || [ "${sockets:-0}" != "$(( TOTAL_CORES / CORES_PER_SOCKET ))" ]; then
    echo "This is not the experimental testbed." >&2
    echo "  expected: ${TOTAL_CORES} cores across $(( TOTAL_CORES / CORES_PER_SOCKET )) sockets, with cpupower" >&2
    echo "  found:    ${online_cores} cores across ${sockets:-?} socket(s)," \
         "cpupower $(command -v cpupower > /dev/null && echo present || echo missing)" >&2
    echo "See the README's 'The testbed' section for what that machine was, and" >&2
    echo "the README's 'What survived' section for what is reproducible without it." >&2
    exit 1
fi

if [ "$N_CORES" -lt 1 ] || [ "$N_CORES" -gt "$MAX_WORKLOAD_CORES" ]; then usage; fi

# Cores are allocated socket-interleaved: 0, 10, 1, 11, 2, 12, ... so that a
# given core count is spread evenly over the two sockets rather than filling
# socket 0 first. Filling one socket first would confound "more cores" with
# "second socket now powered up", which changes the energy picture for reasons
# that have nothing to do with the workload.
#
# This must agree with omp_energy_rl.platform_config.affinity_mask(), which
# generates the same 19 masks the original controller hard-coded.
mask=0
allocated=0
highest=0
for (( i = 0; i < CORES_PER_SOCKET; i++ )); do
    for (( s = 0; s < TOTAL_CORES / CORES_PER_SOCKET; s++ )); do
        core=$(( s * CORES_PER_SOCKET + i ))
        [ "$core" -eq "$CONTROLLER_CORE" ] && continue
        [ "$allocated" -ge "$N_CORES" ] && break 2
        mask=$(( mask | (1 << core) ))
        [ "$core" -gt "$highest" ] && highest=$core
        allocated=$(( allocated + 1 ))
    done
done
MASK=$(printf '0x%05X' "$mask")

# Order matters. Dropping every core to the minimum first means cores that are
# NOT in the allocation do not sit parked at whatever frequency the previous
# configuration left them at, still drawing package power that CpJ would then be
# charged for. Then the controller's own core goes back up, then the allocated
# range is raised.
cpupower --cpu all frequency-set --freq "${MIN_FREQ}GHz"            > /dev/null
cpupower --cpu "$CONTROLLER_CORE" frequency-set --freq "${CONTROLLER_FREQ}GHz" > /dev/null
cpupower --cpu "0-${highest}" frequency-set --freq "${FREQ}GHz"     > /dev/null

echo "config: ${N_CORES} cores @ ${FREQ}GHz -> affinity ${MASK}"

if [ -n "$PID" ]; then
    # -a applies to every thread of the process, not just the main one.
    taskset -pa "$MASK" "$PID" > /dev/null
    echo "pinned pid ${PID} to ${MASK}"
else
    echo "no pid given; launch the workload with: taskset ${MASK} <program> ..."
fi
