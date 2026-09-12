"""The experimental platform's configuration space, as a codec.

The testbed is a dual-socket Intel Xeon E5-2630 v4 server: 20 physical cores,
10 per socket, socket 0 = cores 0-9 and socket 1 = cores 10-19. Core 19 is
reserved for the controller process itself, leaving **19 cores** for the
workload. Operating frequencies run from 1.2 GHz to 2.2 GHz in 100 MHz steps --
**11 frequencies**; anything above 2.2 GHz is excluded because TDP throttling
makes it impossible to hold a requested frequency constant (thesis section 4.3).

19 core counts x 11 frequencies = **209 configurations**, which is exactly the
DQN's action space. This module is the single source of truth for that mapping:

* :func:`action_to_config` / :func:`config_to_action` -- action id <-> (cores, GHz)
* :func:`affinity_mask` -- core count -> CPU affinity mask for ``taskset``

Cores are allocated **socket-interleaved** (0, 10, 1, 11, 2, 12, ...) so that a
given core count is spread evenly across the two sockets rather than filling
socket 0 first. The original controller hard-coded the resulting 19 masks as a
19-branch ``if/elif`` chain; here they are generated from the rule, and
``platform/set_config.sh`` derives the same masks a second time, in shell.

Nothing in this module touches hardware -- it is pure arithmetic, and is
importable anywhere.
"""

from __future__ import annotations

from typing import List, Tuple

#: Physical cores on the testbed, and how they are split between sockets.
TOTAL_CORES = 20
CORES_PER_SOCKET = 10
N_SOCKETS = 2

#: Core 19 (socket 1's last core) runs the controller/collector process itself,
#: pinned at 3.10 GHz so measurement overhead never competes with the workload.
CONTROLLER_CORE = 19
CONTROLLER_FREQ_GHZ = 3.10

#: Cores available to the controlled workload: 1..19.
MAX_WORKLOAD_CORES = TOTAL_CORES - 1

#: Frequency grid: 1.2 .. 2.2 GHz, 100 MHz steps.
MIN_FREQ_GHZ = 1.2
FREQ_STEP_GHZ = 0.1
N_FREQUENCIES = 11

#: Size of the DQN action space.
N_ACTIONS = MAX_WORKLOAD_CORES * N_FREQUENCIES  # 209

#: Controller sampling period (thesis section 4.2.3.3). One action per period.
SAMPLING_PERIOD_S = 0.5


def frequencies_ghz() -> List[float]:
    """The 11 selectable frequencies, in GHz, low to high."""
    return [round(MIN_FREQ_GHZ + FREQ_STEP_GHZ * i, 1) for i in range(N_FREQUENCIES)]


def core_order() -> List[int]:
    """Physical core ids in the order the workload claims them.

    Socket-interleaved: ``[0, 10, 1, 11, 2, 12, ..., 9]``. Core 19 never appears
    -- it belongs to the controller.
    """
    order: List[int] = []
    for i in range(CORES_PER_SOCKET):
        for s in range(N_SOCKETS):
            core = s * CORES_PER_SOCKET + i
            if core != CONTROLLER_CORE:
                order.append(core)
    return order


def cores_for(n_cores: int) -> List[int]:
    """The physical core ids allocated for a workload of ``n_cores`` cores."""
    if not 1 <= n_cores <= MAX_WORKLOAD_CORES:
        raise ValueError(
            f"n_cores must be in 1..{MAX_WORKLOAD_CORES}, got {n_cores}"
        )
    return core_order()[:n_cores]


def affinity_mask(n_cores: int) -> int:
    """CPU affinity bitmask for ``n_cores`` workload cores.

    Pass to ``taskset`` as hex, e.g. ``taskset -pa 0x00C03 <pid>`` for 4 cores.
    """
    mask = 0
    for core in cores_for(n_cores):
        mask |= 1 << core
    return mask


#: Mask for the controller's own core, used as ``taskset 0x80000 python ...``.
CONTROLLER_MASK = 1 << CONTROLLER_CORE


def action_to_config(action: int) -> Tuple[int, float]:
    """Decode a DQN action id into ``(n_cores, freq_ghz)``.

    Matches the original ``setConfiguration``: frequency is the fast-varying
    axis (``action % 11``), core count the slow one (``action // 11``).
    """
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {action}")
    freq = round(MIN_FREQ_GHZ + FREQ_STEP_GHZ * (action % N_FREQUENCIES), 1)
    n_cores = action // N_FREQUENCIES + 1
    return n_cores, freq


def config_to_action(n_cores: int, freq_ghz: float) -> int:
    """Encode ``(n_cores, freq_ghz)`` back into a DQN action id."""
    if not 1 <= n_cores <= MAX_WORKLOAD_CORES:
        raise ValueError(
            f"n_cores must be in 1..{MAX_WORKLOAD_CORES}, got {n_cores}"
        )
    freq_index = round((freq_ghz - MIN_FREQ_GHZ) / FREQ_STEP_GHZ)
    if not 0 <= freq_index < N_FREQUENCIES:
        raise ValueError(
            f"freq_ghz must be one of {frequencies_ghz()}, got {freq_ghz}"
        )
    return (n_cores - 1) * N_FREQUENCIES + freq_index


def normalize_config(n_cores: int, freq_ghz: float) -> Tuple[float, float]:
    """Scale a configuration into ``[0, 1]`` for the agent's state vector.

    Mirrors ``captureState`` in the original controller: frequency is mapped
    from its 1.2-2.2 GHz range and core count from its 1-19 range.
    """
    f = (freq_ghz - MIN_FREQ_GHZ) / (FREQ_STEP_GHZ * (N_FREQUENCIES - 1))
    c = (n_cores - 1) / (MAX_WORKLOAD_CORES - 1)
    return f, c
