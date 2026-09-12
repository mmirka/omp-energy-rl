# Instrumented example applications

Two OpenMP programs that the CpS/CpJ instrumentation and the online controller
were exercised against. Both are **reconstructions**: the original C sources did
not survive the archive (only their invocation scripts did), so these are rebuilt
from the published benchmark template and the recovered command-line contract.
See the main [`README`](../README.md), section "What survived, and what did not".

```bash
make                 # build both with the stock compiler
make smoke           # a few seconds each, confirms they run
make CC=/opt/gcc-4.8.0-chunkcount/bin/gcc    # build instrumented
```

## What "instrumented" means here

Nothing in these programs counts anything. The chunk counter lives in the
**OpenMP runtime**, not in the application: a patched `libgomp` increments it
every time the dynamic scheduler hands a chunk to a thread, and publishes the
running total into shared memory. See the main [`README`](../README.md),
section "Chunk counting in libgomp".

That is the whole point of the approach. An application needs **no source
changes, no annotations and no profiling run** to be measurable — it only needs
to be built against the patched runtime. Built against a stock compiler these two
programs run identically and are simply not observed.

The one thing the application does have to do is use `schedule(dynamic, 1)` on
the loop of interest, so that **one chunk equals one outer iteration**. With a
larger chunk size or a static schedule the counter advances in steps coarser than
the controller's 500 ms sampling period and the CpS signal becomes unusable. The
original binaries carried a `_1chunk` suffix in their names for exactly this
reason.

## `mixed_cxmy` — the parametrizable single-phase benchmark

```
./mixed_cxmy <iterations> <cpu_percent> <nthreads>
```

Thesis Fig. 4.12 / MOCAST 2020 Fig. 2. Each outer iteration runs 100 inner steps;
`cpu_percent` of them run a compute-intensive fragment (a Fourier partial sum
plus a dependent floating-point chain, register-resident) and the rest a
memory-intensive one (scattered read / arithmetic / scattered write over a 256 MiB
working set, randomized to defeat the hardware prefetcher).

The six benchmarks characterized in thesis Table 4.2 are this one program at six
settings:

| Benchmark | Invocation | Best configuration (thesis Table 4.5) | Best CpJ |
|---|---|---|---|
| C100M0 | `./mixed_cxmy 2000000000 100 19` | 19 cores @ 2.1 GHz | 3866 |
| C98M2  | `./mixed_cxmy 2000000000  98 19` | 13 cores @ 2.1 GHz | 2505 |
| C96M4  | `./mixed_cxmy 2000000000  96 19` |  7 cores @ 2.1 GHz | 1581 |
| C90M10 | `./mixed_cxmy 2000000000  90 19` |  3 cores @ 2.1 GHz |  818 |
| C80M20 | `./mixed_cxmy 2000000000  80 19` |  2 cores @ 2.1 GHz |  517 |
| C0M100 | `./mixed_cxmy 2000000000   0 19` |  1 core  @ 1.9 GHz |  336 |

Read that table downwards: as the workload shifts from compute-bound to
memory-bound, the energy-optimal core count collapses from 19 to 1. More cores on
a memory-bound workload buy no throughput — the memory subsystem is already
saturated — while still drawing power, so CpJ falls. **This is the entire
motivation for the work**: a controller that only tunes frequency, as the Linux
governors do, cannot express the decision that matters here.

That the reconstruction reproduces the underlying behaviour is checkable without
any energy measurement, just by looking at how the two extremes scale with thread
count (measured on a 24-thread desktop, 200k iterations):

| threads | 1 | 2 | 4 | 8 | 16 | speedup |
|---|---|---|---|---|---|---|
| C100M0 (compute) | 34k | 68k | 135k | 253k | 357k | **10.5x** |
| C0M100 (memory)  | 297k | 537k | 809k | 980k | 1091k | **3.7x** |

The compute-bound end scales close to linearly; the memory-bound end saturates,
gaining only 11% from 8 to 16 threads. Add power to that picture and the
memory-bound optimum moves to a single core.

## `two_phase` — the multi-phase benchmark

```
./two_phase <iterations> <mem_phase_iters> <cpu_phase_iters> <nthreads>
./two_phase 1000000000 5000000 20000000 19      # the recovered invocation
```

Alternates long stretches of the two fragments. This is the workload behind the
thesis's central multi-phase result (section 4.3.2.2, Figs. 4.23–4.24), and it is
what the shipped phase encoder `encoder_2bit_adac2_2P.h5` was trained on.

Its two phases have deliberately **distant** optima — roughly 4 cores @ 1.3 GHz
for the memory phase, 19 cores @ 2.2 GHz for the compute phase. Which is the
whole experiment: run the identical controller with and without the phase
autoencoder, and the phase-aware version tracks the core count almost exactly (19
and 5 against a known-best 19 and 4) while the phase-blind one shows no
correlation between its actions and the phase changes at all, and loses 34% of
mean CpJ.

The phase is a function of the **iteration index**, not of wall time, so phase
boundaries sit at fixed points in the work regardless of what configuration the
controller picks. The controller cannot move a phase boundary by acting — the
phases are a property of the workload, not of the control policy.

> **Argument order is inferred.** The only surviving evidence is the invocation
> `run_code_2phases 1000000000 5000000 20000000 19` in a recovered launch script.
> Total-iterations / memory-phase-length / compute-phase-length / threads is the
> reading consistent with the sibling scripts, but it is a reading, not a
> recovered signature.

## The third-party benchmarks

Two benchmarks used in the papers are **not** vendored here — they are other
people's code, and this repository ships no third-party sources. Both were used
unmodified apart from being built against the patched toolchain and invoked with
a long iteration count.

**SRAD**, from the [Rodinia](https://rodinia.cs.virginia.edu/) benchmark suite
(`srad_v2`). The real-world multi-phase application of ReCoSoC 2019 and thesis
section 4.3.2.1; its two phases are visible directly in a CpS trace. The recovered
invocation:

```bash
taskset 0x7FFFF ./srad 32768 32768 0 127 0 127 19 0.5 1000000000
```

— a 32768x32768 image, the full window, 19 threads, lambda 0.5, and an iteration
count large enough that the run outlives the controller's 1024 s exploration
period. The Xeon characterization sweep from these runs *is* bundled, at
[`../data/srad_allConf_19c11f.csv.gz`](../data/srad_allConf_19c11f.csv.gz) — it is
what the phase autoencoder trains on.

**DGEMM**, from the
[Parallel Research Kernels](https://github.com/ParRes/Kernels) (`OPENMP/DGEMM`).
The single-phase compute-bound benchmark of thesis section 4.3.1. Recovered
invocation:

```bash
taskset 0x7FFFF ./dgemm 19 1000000000 1024
```

Its full 209-configuration characterization sweep is bundled at
[`../figures/data/dgemm_19c11f.csv`](../figures/data/dgemm_19c11f.csv) and is what
`figures/cpj_surface.py` reduces.
