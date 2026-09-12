"""CpS and CpJ: deriving energy efficiency from the OpenMP chunk counter.

A *chunk* is one iteration of a dynamically scheduled OpenMP parallel loop -- the
unit of work ``libgomp`` hands to a thread. The patched runtime (documented in
the README's 'Chunk counting in libgomp' section) keeps a per-thread chunk tally, sums it, and
publishes ``"<chunks>&<timestamp_ns>"`` into a System V shared-memory segment. A
separate collector process reads that segment periodically, and reads the CPU's
RAPL energy counters alongside it. From two consecutive readings:

.. math::

    CpS = \\frac{\\Delta chunks}{\\Delta t} \\qquad
    CpJ = \\frac{\\Delta chunks}{\\Delta E}

CpS is a performance metric, CpJ an energy-efficiency one. Both are relative to
the parallel loop being measured -- chunk size varies with the loop body, so CpS
and CpJ are *not* comparable across different applications. They are comparable
across configurations of the same application, which is all the controller needs.

This module is the chunk-stream-to-CpS/CpJ fusion step. The original archive
preserved the producer (the ``libgomp`` patch) and the consumers (the analysis
notebooks, which read already-fused CSVs), but not this step -- it is
reconstructed here from the definitions in thesis section 4.1.1.2.

Nothing here talks to hardware; :class:`EfficiencySampler` takes raw readings from
whatever source and is directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Width of the shared-memory payload written by the patched libgomp, in bytes.
#: The C side declares ``char chaineAEcrire[64]`` and formats
#: ``"%010lu&%016llu"`` -- a 10-digit cumulative chunk count, a ``&`` separator,
#: and a 16-digit CLOCK_MONOTONIC timestamp in nanoseconds -- then NUL-pads.
SHM_PAYLOAD_BYTES = 64

#: SysV IPC key the patched libgomp creates its segment under. Hard-coded on
#: both sides of the original implementation; kept here for fidelity.
SHM_KEY = 123456

#: The chunk counter is printed with ``%010lu`` and wraps at 10 decimal digits.
CHUNK_COUNTER_MODULUS = 10 ** 10

#: RAPL energy MSRs are 32-bit and wrap. The original controller did not correct
#: for this, which is why some recovered traces carry negative Joule values (see
#: the README's 'What survived, and what did not' section). :class:`EfficiencySampler` corrects for it.
RAPL_COUNTER_BITS = 32


@dataclass(frozen=True)
class ChunkReading:
    """One reading of the shared-memory chunk counter."""

    chunks: int
    """Cumulative chunks executed since the workload started."""

    timestamp_ns: int
    """``CLOCK_MONOTONIC`` value when the runtime last updated the counter."""


@dataclass(frozen=True)
class Efficiency:
    """A CpS/CpJ pair derived from two consecutive readings."""

    cps: float
    """Chunks per second -- performance."""

    cpj: float
    """Chunks per Joule -- energy efficiency. The controller's reward signal."""

    d_chunks: int
    """Chunks retired between the two readings."""

    d_time_s: float
    """Wall time between the two readings, from the runtime's own timestamps."""

    d_energy_j: float
    """Energy consumed between the two readings, in Joules."""


def parse_chunk_payload(raw: bytes) -> ChunkReading:
    """Parse the raw shared-memory bytes into a :class:`ChunkReading`.

    Expects ``b"<10 digits>&<16 digits>"`` followed by NUL padding. The original
    controller did this by taking ``repr()`` of the bytes and splitting on
    backslashes, quotes, ``&`` and ``\\0`` in sequence; this reads the same wire
    format directly.

    Raises :class:`ValueError` if the segment does not hold a well-formed
    payload -- which happens legitimately before the workload has retired its
    first chunk, and callers should treat as "no data yet".
    """
    text = raw.split(b"\x00", 1)[0].decode("ascii", errors="strict")
    chunks_str, _, time_str = text.partition("&")
    if not chunks_str or not time_str:
        raise ValueError(f"malformed chunk payload: {raw!r}")
    return ChunkReading(chunks=int(chunks_str), timestamp_ns=int(time_str))


def _wrapped_delta(current: int, previous: int, modulus: int) -> int:
    """Difference of two counter values, correcting for a single wraparound."""
    delta = current - previous
    if delta < 0:
        delta += modulus
    return delta


class EfficiencySampler:
    """Turns a stream of (chunk counter, energy counter) readings into CpS/CpJ.

    Stateful by nature: every value is a difference against the previous
    reading, so the first :meth:`update` establishes a baseline and returns
    ``None``.

    Both counters are treated as monotonic-with-wraparound. A reading in which no
    chunk was retired, no time passed, or no energy was recorded yields zeroes
    rather than a division error -- matching the original controller, which
    emitted ``CPS = CPJ = 0`` in exactly those cases and relied on downstream
    outlier filtering.
    """

    def __init__(self, rapl_counter_bits: int = RAPL_COUNTER_BITS) -> None:
        self._energy_modulus = float(2 ** rapl_counter_bits)
        self._prev: Optional[ChunkReading] = None
        self._prev_energy_j: Optional[float] = None

    def update(self, reading: ChunkReading, energy_j: float) -> Optional[Efficiency]:
        """Fold in one reading.

        ``energy_j`` is the cumulative Joules reported by the energy counters,
        already summed over whichever RAPL domains and sockets are of interest
        (see :mod:`omp_energy_rl.rapl`). Returns ``None`` for the first reading.
        """
        prev, prev_energy = self._prev, self._prev_energy_j
        self._prev, self._prev_energy_j = reading, energy_j
        if prev is None or prev_energy is None:
            return None

        d_chunks = _wrapped_delta(reading.chunks, prev.chunks, CHUNK_COUNTER_MODULUS)
        d_time_s = (reading.timestamp_ns - prev.timestamp_ns) / 1e9
        d_energy_j = energy_j - prev_energy
        if d_energy_j < 0.0:
            d_energy_j += self._energy_modulus

        cps = d_chunks / d_time_s if d_chunks and d_time_s > 0.0 else 0.0
        cpj = d_chunks / d_energy_j if d_chunks and d_energy_j > 0.0 else 0.0
        return Efficiency(
            cps=cps,
            cpj=cpj,
            d_chunks=d_chunks,
            d_time_s=d_time_s,
            d_energy_j=d_energy_j,
        )

    def reset(self) -> None:
        """Forget the baseline, e.g. after restarting the measured workload."""
        self._prev = None
        self._prev_energy_j = None
