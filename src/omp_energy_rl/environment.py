"""The reinforcement-learning environment: the physical multicore server.

In RL terms the environment is a dual-socket Intel Xeon E5-2630 v4 server running
an OpenMP workload. Acting on it means changing the configuration -- how many
cores the workload may use, and at what frequency. Observing it means reading the
chunk counter the patched ``libgomp`` publishes and the RAPL energy counters, and
turning the pair into CpS and CpJ.

That environment **is not reproduced here.** The machine is 2016-era hardware that
is no longer available, and the measurements depend on it in ways that cannot be
faked without inventing numbers: the RAPL energy model, the memory-subsystem
saturation behaviour that makes the memory-bound workloads prefer *one* core, and
the specific frequency/thermal envelope. :class:`XeonEnvironment` below is the
real interface, written out in full so the control loop is legible end to end --
but it verifies its preconditions on construction and raises
:class:`HardwareUnavailable` anywhere else. There is deliberately **no simulated
fallback**: a controller silently learning against a made-up environment would
produce plausible-looking numbers that mean nothing.

What the environment needs, all of which the README's 'The testbed' section describes:

* a workload compiled against the patched GCC/``libgomp``, publishing its chunk
  counter into System V shared memory under key ``123456``
  (the README's 'Chunk counting in libgomp' section);
* ``/dev/cpu/*/msr`` readable, for RAPL (:mod:`omp_energy_rl.rapl`);
* ``cpupower`` and ``taskset``, to apply actions;
* the ``acpi-cpufreq`` driver rather than ``intel_pstate``, so that requested
  frequencies are actually held -- and so that the Linux governors used as the
  comparison baseline are available at all;
* a 20-core, 2-socket topology, because the 209-action configuration space is
  defined against it (:mod:`omp_energy_rl.platform_config`).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import List, Optional, Tuple

try:  # pragma: no cover - present only on the testbed
    import sysv_ipc
except ImportError:  # pragma: no cover
    sysv_ipc = None  # type: ignore[assignment]

from . import platform_config as plat
from .metrics import SHM_KEY, SHM_PAYLOAD_BYTES, ChunkReading, Efficiency, EfficiencySampler, parse_chunk_payload
from .rapl import RaplReader, RaplUnavailable


class HardwareUnavailable(RuntimeError):
    """Raised when this machine is not the experimental testbed.

    The message names every unmet precondition at once, so a single attempt
    tells you the whole story rather than one missing piece at a time.
    """


def check_preconditions() -> List[str]:
    """Return a list of unmet preconditions; empty means the testbed is present."""
    problems: List[str] = []

    if sysv_ipc is None:
        problems.append(
            "the 'sysv_ipc' package is not installed -- needed to read the chunk "
            "counter the patched libgomp publishes"
        )
    else:
        try:
            sysv_ipc.SharedMemory(SHM_KEY)
        except Exception as exc:  # sysv_ipc.ExistentialError and friends
            problems.append(
                f"no System V shared-memory segment under key {SHM_KEY} ({exc}). "
                "Start a workload built against the patched GCC/libgomp first; see "
                "the README's 'Chunk counting in libgomp' section"
            )

    try:
        RaplReader()
    except RaplUnavailable as exc:
        problems.append(f"RAPL energy counters unreadable: {exc}")

    for tool in ("cpupower", "taskset"):
        if shutil.which(tool) is None:
            problems.append(f"'{tool}' is not on PATH -- needed to apply actions")

    try:
        sockets = len(_physical_packages())
        cores = _online_core_count()
    except OSError as exc:
        problems.append(f"cannot read CPU topology: {exc}")
    else:
        if cores != plat.TOTAL_CORES or sockets != plat.N_SOCKETS:
            problems.append(
                f"topology mismatch: found {cores} cores across {sockets} socket(s); "
                f"the 209-configuration action space assumes {plat.TOTAL_CORES} cores "
                f"across {plat.N_SOCKETS} sockets (see the README's 'The testbed' section)"
            )

    return problems


def _physical_packages() -> List[int]:
    from .rapl import detect_packages

    return detect_packages()


def _online_core_count() -> int:
    import os

    count = 0
    while os.path.exists(f"/sys/devices/system/cpu/cpu{count}/topology/physical_package_id"):
        count += 1
    return count


class XeonEnvironment:
    """The dual-socket Xeon testbed, as an RL environment.

    Args:
        workload_pid: PID of the OpenMP workload to steer. The controller
            re-pins *that* process; it never moves itself (it lives on core 19,
            pinned by ``platform/launch_control.sh``).
        sampling_period_s: seconds between observations. 500 ms in every
            published experiment -- long enough for a training step and two
            inferences, short enough to track a phase change.

    Raises:
        HardwareUnavailable: if this is not the testbed.
    """

    def __init__(
        self,
        workload_pid: int,
        sampling_period_s: float = plat.SAMPLING_PERIOD_S,
    ) -> None:
        problems = check_preconditions()
        if problems:
            raise HardwareUnavailable(
                "This machine is not the experimental testbed:\n"
                + "\n".join(f"  - {p}" for p in problems)
                + "\n\nThe controller is not simulated: see the README's 'The testbed' section for what "
                "the environment was, and the README's 'What survived, and what did not' section for what remains "
                "reproducible without it (the phase autoencoder, the "
                "characterization analysis, and the figures)."
            )

        self.workload_pid = workload_pid
        self.sampling_period_s = sampling_period_s
        self._shm = sysv_ipc.SharedMemory(SHM_KEY)
        self._rapl = RaplReader()
        self._sampler = EfficiencySampler()
        self._n_cores = 1
        self._freq_ghz = plat.MIN_FREQ_GHZ
        self._last_sample_at = time.time()

    # -- observation -------------------------------------------------------- #

    @property
    def configuration(self) -> Tuple[int, float]:
        """The configuration currently applied, as ``(n_cores, freq_ghz)``."""
        return self._n_cores, self._freq_ghz

    def read_chunks(self) -> Optional[ChunkReading]:
        """Read the chunk counter, or ``None`` if the workload has yet to write."""
        try:
            return parse_chunk_payload(self._shm.read(SHM_PAYLOAD_BYTES))
        except ValueError:
            return None

    def wait_for_period(self) -> float:
        """Sleep out the remainder of the current sampling period.

        Returns the time actually slept -- negative-clamped to zero, so an
        overrun (a training step that took longer than the period) shows up as
        ``0.0`` rather than shifting the schedule. The recovered traces log this
        value, which is how the overruns are visible in them at all.
        """
        elapsed = time.time() - self._last_sample_at
        remaining = max(0.0, self.sampling_period_s - elapsed)
        time.sleep(remaining)
        self._last_sample_at = time.time()
        return remaining

    def sample(self) -> Optional[Efficiency]:
        """Take one CpS/CpJ measurement. ``None`` until a baseline exists."""
        reading = self.read_chunks()
        if reading is None:
            return None
        return self._sampler.update(reading, self._rapl.read().total)

    # -- action ------------------------------------------------------------- #

    def apply_action(self, action: int) -> Tuple[int, float]:
        """Apply DQN action ``action``; returns the resulting configuration.

        Three shell calls, in the order the original used and for the reason it
        used them:

        1. drop *every* core to the minimum frequency, so cores dropped from the
           allocation do not stay parked at a high frequency and keep burning
           power that CpJ would then be charged for;
        2. restore the controller's own core to its pinned high frequency, so
           measurement never becomes the bottleneck;
        3. raise the allocated cores to the requested frequency.

        Then re-pin the workload with ``taskset``.
        """
        n_cores, freq_ghz = plat.action_to_config(action)
        self.set_configuration(n_cores, freq_ghz)
        return n_cores, freq_ghz

    def set_configuration(self, n_cores: int, freq_ghz: float) -> None:
        """Apply ``(n_cores, freq_ghz)`` to the machine and the workload."""
        allocated = plat.cores_for(n_cores)
        highest = max(allocated)

        self._run(f"cpupower --cpu all frequency-set --freq {plat.MIN_FREQ_GHZ:.2f}GHz")
        self._run(
            f"cpupower --cpu {plat.CONTROLLER_CORE} frequency-set "
            f"--freq {plat.CONTROLLER_FREQ_GHZ:.2f}GHz"
        )
        self._run(f"cpupower --cpu 0-{highest} frequency-set --freq {freq_ghz:.2f}GHz")
        self._run(f"taskset -pa 0x{plat.affinity_mask(n_cores):05X} {self.workload_pid}")

        self._n_cores = n_cores
        self._freq_ghz = freq_ghz

    @staticmethod
    def _run(command: str) -> None:
        """Fire a shell command without waiting for it.

        Not waiting is deliberate and is what the original did: ``cpupower`` and
        ``taskset`` take tens of milliseconds, and the control loop's budget is
        500 ms shared with a training step. The configuration lands well before
        the next measurement window opens.
        """
        subprocess.Popen(command, shell=True, executable="/bin/bash")

    def describe(self) -> str:
        return (
            f"XeonEnvironment(pid={self.workload_pid}, "
            f"period={self.sampling_period_s}s)\n  {self._rapl.describe()}\n"
            f"  action space: {plat.N_ACTIONS} configurations "
            f"({plat.MAX_WORKLOAD_CORES} core counts x {plat.N_FREQUENCIES} frequencies)"
        )
