"""Reading Intel RAPL energy counters straight from the MSRs.

RAPL (Running Average Power Limit) exposes cumulative energy counters per socket
for several domains. The controller needs two of them:

* **PKG** -- the whole processor package.
* **DRAM** -- the attached memory. Not optional here: the workloads under study
  span the compute-bound/memory-bound spectrum, and for the memory-bound end
  DRAM is where a large share of the energy goes. Omitting it would bias CpJ
  toward the compute-bound configurations.

Energy is summed over PKG + DRAM across both sockets, which is what the recovered
controller did and what the published CpJ figures are computed from.

Why raw MSRs rather than a library: reading ``/dev/cpu/N/msr`` with ``pread`` is a
handful of microseconds, small enough to sit inside a 500 ms control loop
alongside a network inference and a training step. The alternative used earlier
in the project -- shelling out to the Intel PCM tool -- cost far more, and PCM's
native wrapper library did not survive the archive in any case (see
the README's 'What survived, and what did not' section).

**This module needs root and a real Intel CPU.** ``/dev/cpu/*/msr`` requires the
``msr`` kernel module and ``CAP_SYS_RAWIO``; it is routinely unavailable inside
containers, VMs and cloud instances. Every entry point raises
:class:`RaplUnavailable` with the specific missing precondition rather than
returning wrong numbers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# --- MSR addresses (Intel SDM Vol. 4) -------------------------------------- #

MSR_RAPL_POWER_UNIT = 0x606
MSR_PKG_ENERGY_STATUS = 0x611
MSR_PP0_ENERGY_STATUS = 0x639
MSR_DRAM_ENERGY_STATUS = 0x619

#: CPUID model number of the testbed's Xeon E5-2630 v4.
CPU_BROADWELL_EP = 79

#: Model -> name, for the subset relevant to this project. RAPL semantics --
#: notably whether DRAM uses its own energy unit -- vary across these.
KNOWN_MODELS: Dict[int, str] = {
    45: "Sandy Bridge-EP",
    62: "Ivy Bridge-EP",
    63: "Haswell-EP",
    79: "Broadwell-EP",
    85: "Skylake-X",
    87: "Knights Landing",
}

#: On Haswell-EP and later server parts the DRAM domain uses a fixed 15.3 uJ
#: unit rather than the package's programmable one. Getting this wrong scales
#: every DRAM reading by a constant factor.
DRAM_FIXED_ENERGY_UNIT = 0.5 ** 16


class RaplUnavailable(RuntimeError):
    """Raised when RAPL cannot be read on this machine.

    Carries the specific missing precondition -- no ``msr`` device, no
    permission, unknown CPU -- so the caller can report it rather than silently
    producing zeros.
    """


@dataclass(frozen=True)
class EnergySample:
    """Cumulative Joules per socket, per domain, at one instant."""

    package: List[float]
    dram: List[float]
    pp0: List[float]
    """PowerPlane0 (cores only). Read for completeness; the published CpJ uses
    package + DRAM. Reads as zero on some Haswell/Broadwell server parts."""

    @property
    def total(self) -> float:
        """PKG + DRAM summed over every socket -- the CpJ denominator."""
        return sum(self.package) + sum(self.dram)


def detect_cpu_model() -> int:
    """Read the CPU model number from ``/proc/cpuinfo``.

    The recovered implementation was a stub that ignored the machine entirely and
    returned Broadwell-EP unconditionally -- harmless on the one machine it ran
    on, wrong anywhere else. This reads the real value.
    """
    try:
        with open("/proc/cpuinfo", "r") as handle:
            text = handle.read()
    except OSError as exc:  # pragma: no cover - depends on the host OS
        raise RaplUnavailable(f"cannot read /proc/cpuinfo: {exc}") from exc

    match = re.search(r"^model\s*:\s*(\d+)", text, re.MULTILINE)
    if match is None:
        raise RaplUnavailable("no 'model' field in /proc/cpuinfo; not an x86 CPU?")
    return int(match.group(1))


def detect_packages() -> List[int]:
    """One representative core id per physical socket, in socket order.

    RAPL counters are per-socket, so exactly one core per socket needs its MSR
    read.
    """
    representatives: Dict[int, int] = {}
    cpu = 0
    while True:
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        if not os.path.exists(path):
            break
        with open(path, "r") as handle:
            package = int(handle.read())
        representatives.setdefault(package, cpu)
        cpu += 1
    if not representatives:
        raise RaplUnavailable(
            "no CPU topology under /sys/devices/system/cpu -- cannot locate sockets"
        )
    return [representatives[p] for p in sorted(representatives)]


def _open_msr(core: int) -> int:
    path = f"/dev/cpu/{core}/msr"
    if not os.path.exists(path):
        raise RaplUnavailable(
            f"{path} does not exist. Load the kernel module with 'modprobe msr'; "
            "note it is typically unavailable inside containers and VMs."
        )
    try:
        return os.open(path, os.O_RDONLY)
    except PermissionError as exc:
        raise RaplUnavailable(
            f"no permission to read {path}; RAPL MSR access needs root or "
            "CAP_SYS_RAWIO."
        ) from exc


def _read_msr(fd: int, address: int) -> int:
    return int.from_bytes(os.pread(fd, 8, address), "little")


class RaplReader:
    """Per-socket RAPL energy reader.

    Construction probes the machine and raises :class:`RaplUnavailable` if RAPL
    cannot be read here -- so an unusable reader never reaches the control loop.

    Args:
        strict_model: when true (the default), refuse to run on a CPU whose RAPL
            semantics this module has not been checked against, rather than
            guessing at the DRAM energy unit.
    """

    def __init__(self, strict_model: bool = True) -> None:
        self.model = detect_cpu_model()
        if self.model not in KNOWN_MODELS:
            message = (
                f"CPU model {self.model} is not in the checked set "
                f"({sorted(KNOWN_MODELS)}); RAPL domain availability and the DRAM "
                "energy unit may differ."
            )
            if strict_model:
                raise RaplUnavailable(message + " Pass strict_model=False to override.")
        self.model_name = KNOWN_MODELS.get(self.model, f"unknown (model {self.model})")

        self.package_cores = detect_packages()
        self.n_packages = len(self.package_cores)
        self.cpu_energy_units: List[float] = []
        self.dram_energy_units: List[float] = []

        for core in self.package_cores:
            fd = _open_msr(core)
            try:
                unit_word = _read_msr(fd, MSR_RAPL_POWER_UNIT)
            finally:
                os.close(fd)
            # Bits 12:8 hold the energy-unit exponent: unit = 0.5 ** exponent J.
            self.cpu_energy_units.append(0.5 ** ((unit_word >> 8) & 0x1F))
            self.dram_energy_units.append(DRAM_FIXED_ENERGY_UNIT)

    def read(self) -> EnergySample:
        """Read the current cumulative energy counters.

        The counters are 32-bit and wrap; differencing and wrap correction are
        :class:`omp_energy_rl.metrics.EfficiencySampler`'s job.
        """
        package: List[float] = []
        dram: List[float] = []
        pp0: List[float] = []
        for index, core in enumerate(self.package_cores):
            fd = _open_msr(core)
            try:
                package.append(
                    _read_msr(fd, MSR_PKG_ENERGY_STATUS) * self.cpu_energy_units[index]
                )
                pp0.append(
                    _read_msr(fd, MSR_PP0_ENERGY_STATUS) * self.cpu_energy_units[index]
                )
                dram.append(
                    _read_msr(fd, MSR_DRAM_ENERGY_STATUS) * self.dram_energy_units[index]
                )
            finally:
                os.close(fd)
        return EnergySample(package=package, dram=dram, pp0=pp0)

    def describe(self) -> str:
        return (
            f"{self.model_name}: {self.n_packages} socket(s), representative cores "
            f"{self.package_cores}, package energy unit "
            f"{self.cpu_energy_units[0]:.3e} J, DRAM energy unit "
            f"{self.dram_energy_units[0]:.3e} J"
        )


def probe() -> Optional[str]:
    """Return ``None`` if RAPL is usable here, else why it is not."""
    try:
        return None if RaplReader() else None
    except RaplUnavailable as exc:
        return str(exc)
