# omp-energy-rl — Online RL control of OpenMP energy efficiency

Optimizing energy efficiency means choosing how much hardware to give a program
and how fast to run it. On a 20-core server that is 209 choices, the best one
depends on the program, it changes *within* a program's execution, and Linux's
own governors can only express a fraction of it — they tune frequency, and have
no say at all over how many cores a parallel workload gets.

This is the code from **Chapter 4 of the thesis** (ReCoSoC 2019, GDR SOC² 2019,
MOCAST 2020): measure a running OpenMP program's energy efficiency online with no
prior profiling, detect its execution phases without being told how many there
are, and let a reinforcement-learning controller pick the configuration.

Three pieces, stacked:

**1. A metric that comes from the programming model, not the hardware.** OpenMP's
dynamic scheduler hands *chunks* of a parallel loop to threads. Count them, and
you get a direct measure of application progress that needs no profiling pass and
no architecture-specific performance counters. Divide by time for **CpS** (chunks
per second, performance); divide by energy for **CpJ** (chunks per Joule,
efficiency). A patched GCC 4.8.0 `libgomp` does the counting, so applications need
**no source changes at all**.

**2. An autoencoder that finds execution phases, unsupervised.** It reconstructs
`[CpS, frequency, core count]` through a bottleneck that is forced to be
*discrete* — a vector of ±1, so each value names a phase. The trick is that the
configuration is re-injected at the decoder: the bottleneck gains nothing by
encoding it, so what it must carry is the part of CpS the configuration does not
explain, which is the workload's phase. The number of phases is never supplied.

**3. A Q-network that chooses configurations online, while the program runs.**
State is the configuration plus the phase code; action is one of the 209
configurations; reward is the resulting CpJ. It learns during the run itself —
2048 sampling periods of exploration, about 17 minutes — on a workload it has
never seen, with no characterization sweep.

CpJ over all 209 configurations has a clear optimum, and *which* optimum depends
sharply on the workload. Compute-bound: 19 cores. Memory-bound: **one core** — the
memory subsystem is already saturated, so the other 18 add no throughput and
plenty of power. The frequency-only axis the Linux governors control cannot get
there from here. That picture is thesis Fig. 4.14/4.20, rebuilt by
[`figures/cpj_surface.py`](figures/cpj_surface.py).

## Layout

```
train_autoencoder.py   train the phase autoencoder on the bundled SRAD sweep
run_controller.py      the online DQN controller (testbed only)
src/omp_energy_rl/     the package: config codec, CpS/CpJ derivation, the phase
                       autoencoder, the DQN agent, the control loop, and the
                       hardware-gated testbed environment and RAPL reader
data/                  the SRAD characterization sweep + its schema
figures/               self-contained: code, data and the papers' originals
platform/             the shell layer: apply a configuration, launch a run
bench/                 the two instrumented example applications, in C/OpenMP
results/               training and controller output (git-ignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate omp-energy-rl
```

No GPU is needed anywhere: the autoencoder trains in about a minute on CPU and
the Q-network is smaller still. `figures/` needs only numpy, pandas and
matplotlib — not TensorFlow.

## Train the phase autoencoder

```bash
python train_autoencoder.py                             # 2-bit code, 210k samples
python train_autoencoder.py --code-size 1               # the ReCoSoC 2019 variant
python train_autoencoder.py --epochs 2 --max-rows 20000 # quick smoke test
python train_autoencoder.py --plot                      # + the phase overlay figure
```

Architecture, from thesis Table 4.4 / MOCAST 2020:

```
3 → 100/100/100 → BinaryDense(code_size) → BatchNorm → binary_tanh
  → concat(frequency, core count) → 100/100/100 → 3
```

MSE, Adam at 1e-5, batch 512, 50 epochs. Every activation is linear except the
bottleneck's `binary_tanh` — the discreteness is the whole mechanism, not the
depth. The bottleneck is a `BinaryDense` layer (vendored BinaryNet, Courbariaux
et al. 2016): weights are binarized on every forward pass while the stored
weights stay real-valued, and gradients flow through a straight-through
estimator.

The input is the bundled SRAD characterization sweep — 209 configurations ×
1000 samples, 210 000 rows. See [`data/README.md`](data/README.md) for its
schema.

What comes out, on the bundled sweep (50 epochs, ~30 s on CPU): the 2-bit code
resolves into **two dominant phases** — 61% of samples at mean CpS 5713, 38% at
mean CpS 10181 — plus two codes that between them claim 0.5% of the trace. That
is SRAD's two-phase structure, resolved without the model ever being told how
many phases there are. `--plot` draws it: in the default window one code sits on
the baseline and the other on the peaks.

One caveat visible in the same figure. The default window is the middle of the
sweep; a window near the *start* sits in the 1-core configuration blocks, where
CpS never leaves the bottom of the global range and every sample collapses into
one code. The separation is real but it is not uniform across the action space.

The script writes **an encoder and its scaler, together**, into
`results/autoencoder/`. They must stay together: the phase code is a function of
min-max-scaled inputs, so the same CpS scaled against a different trace's range
lands in a different phase. The scaler for the original published checkpoint is not included here, which is
exactly why this is now explicit.

Phase codes are an **enumerated type, not an ordering**. The values are arbitrary
and differ between training runs (thesis 4.1.3.2).

## Run the controller

```bash
python run_controller.py --check    # is this machine the testbed?
```

On the testbed, with a workload already running:

```bash
sudo ./platform/launch_control.sh ./bench/two_phase 1000000000 5000000 20000000 19
```

Anywhere else, `--check` lists every unmet precondition — shared-memory segment,
RAPL, tooling, topology — and exits 1.

The agent ships with **three** hyperparameter presets, not one: what thesis
Table 4.3 documents, and the configurations the two original controller
implementations ran. Each preset carries its own `provenance` string, so a run
records which one it used.

Worth knowing before reading the agent: `GAMMA = 0` in every configuration used,
so the Bellman update collapses to `Q[s, a] ← r`. This is deliberate and the
thesis says so — the DQN machinery generalizes over a large non-discrete state
space, it does not assign credit over time.

## Figures

```bash
cd figures
python cpj_surface.py          # Figs 4.14, 4.20 — CpJ over all 209 configurations
python governor_comparison.py  # Tables 4.2, 4.5 — vs the four Linux governors
python controller_trace.py     # Figs 4.18-4.24 — CpJ and actions over a run
python phase_overlay.py        # Fig 4.11 — CpS coloured by detected phase
```

`figures/` is self-contained: it imports nothing from `src/`, reads nothing
outside `figures/data/`, and needs no hardware and no training. The originals as
printed are in `figures/from_papers/` for comparison. See
[`figures/README.md`](figures/README.md).

## Benchmarks

```bash
make -C bench
./bench/mixed_cxmy 2000000000 100 19     # C100M0, compute-bound
./bench/mixed_cxmy 2000000000   0 19     # C0M100, memory-bound
./bench/two_phase 1000000000 5000000 20000000 19
```

Reconstructions of the two synthetic benchmarks, built from the published template.
The duality they exist to demonstrate is checkable without any energy
measurement: over 1→16 threads the compute-bound extreme scales **10.5x** while
the memory-bound extreme saturates at **3.7x**. See
[`bench/README.md`](bench/README.md), which also documents SRAD (Rodinia) and
DGEMM (Parallel Research Kernels) — used in the papers, not vendored here.

---

# CpS and CpJ

Energy efficiency is *work per unit energy*. In HPC that is usually FLOPS/W; in
embedded computing, MIPS/W. Both need a measure of "work", and both get it from
hardware performance counters — which is where the trouble starts. Counters are
architecture-specific, so a metric built on `INST_RETIRED` and `L3_MISS` does not
transfer from a Xeon to an ARM big.LITTLE without being rebuilt. Counters measure
*events*, not progress: instructions retired tells you the machine was busy, not
how much of the application's job got done. And deriving efficiency from
execution time requires knowing the total, which means profiling the application
first — ruling out online control of anything not already characterized.

So take the measure of work from the **programming model** instead. OpenMP's
dynamic loop scheduler divides a parallel loop into *chunks* and hands them to
threads as those threads become free. A chunk is a unit of the application's own
work, defined by the application's own loop.

> **CpS — Chunks per Second.** A *performance* metric.
>
> **CpJ — Chunks per Joule.** An *energy-efficiency* metric. Equivalently CpS
> per Watt.

## The formalization

Thesis section 4.1.1.2. A parallel workload `Par` is a set of chunks `C = {i_k}`
of instruction instances, spread over threads `T = {th_1 … th_n}` with `q_k` the
chunks assigned to thread `th_k` and `q_1 ∪ … ∪ q_n = Par`. Then execution time
`δ_Par = max_k δ_th_k` (the slowest thread sets the wall time) and energy
`ε_Par = Σ_k ε_th_k` (every thread's energy adds up), giving

```
Perf(Par)       = (1 / δ_Par) · Σ_k |q_k|      (thesis 4.1)   → CpS
EnergyEff(Par)  = (1 / ε_Par) · Σ_k |q_k|      (thesis 4.2)   → CpJ
```

`omp_energy_rl.metrics.EfficiencySampler` implements the difference form: two
readings of the chunk counter and the energy counters give `Δchunks / Δt` and
`Δchunks / ΔE`.

## The one thing to keep in mind

**A chunk's size depends on the loop it comes from**, so CpS and CpJ have no
absolute scale. A CpJ of 3866 for one benchmark and 336 for another says nothing
about which is more efficient — the chunks are not the same size.

They are comparable **across configurations of the same workload**, which is
exactly, and only, what the controller needs: it asks "is 8 cores at 1.9 GHz
better than 4 cores at 2.1 GHz *for this program*", never "is this program better
than that one". Every number here should be read that way. Two consequences show
up in the code: `ControllerConfig.cps_scale` is workload-specific and has to be
re-derived per application, and a phase encoder is only valid together with the
`FeatureScaler` fitted alongside it.

## Where the numbers come from

**Chunks** come from the patched OpenMP runtime (below). The application needs no
source change; it only needs its parallel loop to use `schedule(dynamic, 1)`, so
one chunk equals one iteration and the counter advances finely enough to sample
every 500 ms.

**Energy** comes from whatever the platform offers. On the Intel testbed, the
RAPL model-specific registers, package + DRAM, summed over both sockets
(`omp_energy_rl.rapl`). On the Odroid XU3 used in ReCoSoC 2019, the board's own
current sensors. That substitution is the portability argument: the chunk half of
the metric did not change between the two platforms, only the energy half.

---

# Chunk counting in libgomp

> **Documented, not reproduced.** This repository ships no GCC source, no patch
> file and no build system for one. What follows describes the modification
> precisely enough to re-derive it.

`libgomp` is GCC's implementation of the OpenMP runtime, and its dynamic loop
scheduler is what hands chunks to threads. The modification adds a counter there:
every dispatch adds that chunk's iteration count to a per-thread tally, and the
running total is periodically published — with a timestamp — into a System V
shared-memory segment. A collector process reads that segment and differences it.

The consequence that matters: **an application needs no source changes, no
annotations and no profiling run to be measured.** Instrumentation lives one
layer below the application, in the runtime it already links against.

## Where it goes

One file, `libgomp/iter.c`, and two functions in it: `gomp_iter_dynamic_next`
(the lock-free fast path, claiming a chunk with `__sync_fetch_and_add` on
`ws->next`) and `gomp_iter_dynamic_next_locked` (the same under the work-share
lock). Both are patched, so the counter advances either way. Five globals are
added: `nb_chunk_done[24]` (per-thread tally), `nb_chunk_updated` (the published
running total), `maj_done` (a publish-in-progress flag), `shmid` and
`shared_memory`.

Each dispatch adds the chunk's **iteration count**, not 1 — the counter tracks
work, not scheduler activity:

```c
int tid = omp_get_thread_num();
nb_chunk_done[tid] += (nend - tmp);   /* lock-free path: iterations granted */
nb_chunk_done[tid] += chunk;          /* locked path:    the nominal size   */
```

With `schedule(dynamic, 1)` both are 1 per dispatch and the difference
disappears — one more reason the benchmarks pin the chunk size to 1.

Only thread 0 publishes, and only when no publish is in flight. It drains every
thread's tally into `nb_chunk_updated` and writes a 64-byte payload holding
`"<10 digits>&<16 digits>"` then NUL padding — cumulative chunks, `&`, a
`CLOCK_MONOTONIC` timestamp in nanoseconds — into a SysV segment at key `123456`.
`omp_energy_rl.metrics.parse_chunk_payload` reads exactly this. The timestamp is
published *alongside* the counter because the two must be consistent:
`Δchunks / Δt` is only correct if `Δt` is the interval over which those chunks
were counted.

There is no mutex, by deliberate design (thesis 4.1.1.1): collisions are rare and
the resulting outliers are filtered downstream.

## Three defects worth fixing rather than reproducing

1. **A hard-coded 24-thread bound** on `nb_chunk_done`.
2. **The descending-loop branch carries the ascending branch's comparisons**
   (`if (tmp >= end)`, `if (nend > end)`). Any `#pragma omp for` counting *down*
   would be scheduled incorrectly. Every benchmark here counts up, so it never
   fired — but on a general-purpose toolchain it is a correctness bug affecting
   programs that have nothing to do with the measurement.
3. **`shmdt(&shmid)` detaches the wrong address** (should be
   `shmdt(shared_memory)`), so the call fails with `EINVAL` every time and the
   segment stays attached. Harmless here, but the detach never happened.

Two further limits are design choices, not bugs: the IPC key is hard-coded on
both sides, so one instrumented workload at a time; and the counter is printed
with `%010lu`, so it wraps at 10 decimal digits (`EfficiencySampler` corrects for
that wrap; the original did not).

## Rebasing it onto a current GCC

There is no patch file — only the modified `iter.c` in full, against GCC 4.8.0
(2013). Rebasing means re-deriving the instrumentation by hand: `struct
gomp_work_share`, `struct gomp_thread` and the scheduler's fast paths have all
changed, and current versions carry non-monotonic and `doacross` paths that did
not exist in 4.8. The shape to preserve is the one above — weight by iteration
count, accumulate per thread, publish from one thread only, keep the timestamp
alongside the counter.

---

# The testbed

> **Documented, not reproduced.** In reinforcement-learning terms this machine
> *is* the environment. It is 2016-era hardware and is no longer available.
> Nothing here simulates it — `omp_energy_rl.environment` refuses to construct
> off-testbed rather than falling back to anything invented. A controller
> silently learning against a made-up environment would produce
> plausible-looking numbers that mean nothing.

| | |
|---|---|
| CPU | 2 × Intel Xeon E5-2630 v4 (Broadwell-EP, CPUID model 79) |
| Topology | dual socket, 10 cores per socket, **20 cores total** |
| Sockets in `/sys` | socket 0 = cores 0–9, socket 1 = cores 10–19 |
| Frequency driver | `acpi-cpufreq` — **not** `intel_pstate` |
| Frequency range used | 1.2 – 2.2 GHz in 100 MHz steps (**11 frequencies**) |
| Energy measurement | RAPL MSRs via `/dev/cpu/*/msr`, PKG + DRAM, both sockets |
| Compiler | GCC 4.8.0 with the patched `libgomp` |

**Why `acpi-cpufreq`.** `intel_pstate` treats a requested frequency as a hint and
manages P-states itself. The experiment needs a requested frequency to actually
be *held*, or the action the agent took is not the action the machine performed
and the reward is attributed to the wrong thing. `acpi-cpufreq` also exposes the
four classic governors, which are the comparison baseline for the whole chapter.

**Why the ceiling is 2.2 GHz.** The parts go higher, but above 2.2 GHz the TDP
envelope starts throttling and a requested frequency can no longer be guaranteed
constant (thesis 4.3). An action whose effect depends on the chip's thermal
history is not a usable action, so the range stops where reproducibility does.

## The 209 configurations

**19 core counts × 11 frequencies = 209**, and that number is the DQN's action
space. `omp_energy_rl.platform_config` is the single source of truth:

```python
freq_ghz = 1.2 + 0.1 * (action % 11)     # frequency is the fast axis
n_cores  = action // 11 + 1              # core count the slow one
```

**Why 19 cores and not 20.** Core 19 is reserved for the controller and pinned at
3.10 GHz. The controller runs a Keras training step and two inferences inside
every 500 ms period; sharing cores with the workload would perturb the very CpS
and CpJ it is measuring. `platform/launch_control.sh` enforces the split:
workload on `0x7FFFF` (cores 0–18), controller on `0x80000` (core 19).

**Why cores are allocated socket-interleaved.** The order is `0, 10, 1, 11, 2,
12, …`, never filling socket 0 first. Filling one socket first would confound
"the workload got more cores" with "the second socket just came under load", a
step change in package power that has nothing to do with the workload.
Interleaving keeps the sockets symmetric at every core count, so the core-count
axis measures what it claims to. The allocations are also **nested**: growing the
core count only adds cores, never moves existing ones, so a configuration change
never migrates a running thread off a warm cache. The masks this produces are
exactly the 19 the original controller hard-coded — `0x00001, 0x00401, 0x00403,
0x00C03, …, 0x7FFFF` — and `platform/set_config.sh` derives them independently in
shell.

## Applying an action

Four calls, and the order is load-bearing:

```bash
cpupower --cpu all frequency-set --freq 1.20GHz     # 1. everything to the floor
cpupower --cpu 19  frequency-set --freq 3.10GHz     # 2. controller core back up
cpupower --cpu 0-N frequency-set --freq <target>    # 3. allocated cores to target
taskset -pa 0x<mask> <workload_pid>                 # 4. re-pin the workload
```

Floor first, because cores dropped from the allocation must not stay parked at
the previous configuration's frequency — they would keep drawing package power
that CpJ is then charged for. `taskset -pa`, because `-a` covers every thread of
the process; without it the OpenMP workers keep their old affinity and the action
does nothing at all.

## Energy measurement

The controller reads PKG and DRAM on both sockets and sums all four. **DRAM is
not optional**: the workloads span the compute-bound/memory-bound spectrum, and
at the memory-bound end a large share of the energy is in DRAM. Leaving it out
would systematically bias CpJ toward compute-bound configurations — exactly the
wrong direction for the result the chapter is establishing.

Two implementation notes: raw `pread` on `/dev/cpu/N/msr` costs microseconds,
where an earlier Intel PCM-based collector did not (and the PCM native wrapper
is not included here anyway); and on Haswell-EP and later server parts the
DRAM domain uses a fixed 15.3 µJ unit rather than the package's programmable one,
which if got wrong scales every DRAM reading by a constant factor. Reading these
MSRs needs root or `CAP_SYS_RAWIO` and the `msr` module, and is routinely
unavailable in containers, VMs and cloud instances — one reason this environment
cannot simply be moved to rented hardware.

## The sampling period

500 ms, squeezed from both sides. Not shorter than the controller's own work — a
training step (~100 ms measured), a phase inference (< 5 ms), an action inference
(< 2 ms) — plus enough time for the chunk counter to advance measurably. Not
longer than the timescale on which the workload's phases change, or the
controller reacts to a phase that has already ended. The thesis notes this as a
real constraint on which applications are eligible for this kind of control at
all (4.3.2.3).

---

# What is available, what is not

This repository is built from the original PhD working tree — the CpS/CpJ
instrumentation and the phase autoencoder, the DQN controller and its
experimental logs — together with ReCoSoC 2019, GDR SOC² 2019, MOCAST 2020, and
Chapter 4 of the thesis.

## What runs here

| | |
|---|---|
| Phase autoencoder — build, train, infer | ✅ runs on the bundled sweep, CPU, ~1 minute |
| All four figure scripts | ✅ run on bundled data, no training, no hardware |
| Both benchmark applications | ✅ build and run with any OpenMP compiler |
| Config codec, CpS/CpJ derivation, DQN agent | ✅ importable |
| The online control loop, end to end | ❌ needs the testbed |
| Chunk counting | ❌ needs the patched GCC 4.8.0 |
| The published percentages, re-derived from scratch | ❌ quoted and attributed, not regenerated |

## Reproduction checks

Two published numbers come back out of the bundled data.

**DGEMM's best configuration.** Reducing the bundled 209-configuration sweep
gives **19 cores @ 2.1 GHz, mean CpJ 166.2**. Thesis 4.3.1 reports 18 cores @
2.1 GHz, mean CpJ 166. The frequency and the efficiency value reproduce; the core
count lands one apart, within the run-to-run variation the thesis itself notes
for this workload.

**Gains over the Linux governors on DGEMM:**

| vs governor | thesis §4.3.1 | recomputed here |
|---|---|---|
| Performance | 10% | 11.7% |
| Powersave | 17% | 18.4% |
| Ondemand | 11% | 12.3% |
| Conservative | 10% | 12.1% |

Both are consistency checks on the bundled data and the reimplemented analysis,
not independent replications — the underlying measurements are the original ones.

> **Paper-reported (not regenerated here).** Up to **469% CpJ** gain over Linux
> governors on the memory-bound synthetic benchmark; **67%** on a two-phase
> workload; within **3%** of the best-known configuration on DGEMM. Removing the
> phase autoencoder from an otherwise identical two-phase experiment costs
> **34%** of mean CpJ.

## The three agent presets

`omp_energy_rl.agent` exposes three named hyperparameter presets, each with its
own `provenance` string, rather than collapsing them into one.

**The networks differ.** `thesis-table-4.3` is `3 → 8 / 64 / 256 → 209`, all
linear, as documented in the thesis. `recovered-no-autoencoder` has those exact
widths with **ReLU** activations. `recovered-with-autoencoder` is a different
network again: `4 → 128 / 64 / 32 / 16 → 209`, linear, batch size 1024 against
the other's 8, and a 0.05 floor under epsilon against the other's 0. Pick the one
that matches the experiment you are reproducing.

**Exploration stops hard.** `Agent.act` uses the condition
`random() < epsilon **and** steps < observe_period`. The conjunction — rather
than the disjunction a textbook epsilon-greedy uses — means exploration ends
abruptly at 2048 steps and the decayed epsilon afterwards has no effect on
behaviour. The disjunctive form sits alongside it as a commented-out line, so
this was a deliberate switch to a clean explore-then-exploit split. Kept, because
it is what produced the published results.

**It is not multi-step RL.** `GAMMA = 0` in every configuration used, so the
Bellman update collapses to `Q[s, a] ← r`. The thesis says so plainly (4.2.3: the
network "is only trained to perform combinatorial inference […] its decisions
depend only on the current state"). `gamma` remains a parameter because the
implementation supports the general case.

## Not included

**The synthetic benchmark sources.** `bench/mixed_cxmy.c` and `bench/two_phase.c`
are reconstructions from the published template (thesis Fig. 4.12) plus the CLI
contract the original invoking scripts pin down; `two_phase`'s argument
*meanings* are inferred from one such invocation.

**The chunk-stream-to-CpS/CpJ fusion step.** The producer (the `libgomp` patch)
and the consumers (notebooks reading already-fused CSVs) are both here; the step
between them is not. `omp_energy_rl.metrics` implements it from the definitions
in thesis 4.1.1.2.

**The scaler for the published phase encoder.** The `.h5` is here; the
`MinMaxScaler` its inputs were fitted under is not, nor is the trace it was
fitted on. An encoder without its scaler is not usable for its original purpose,
which is why `data.FeatureScaler` is savable and why `train_autoencoder.py`
writes the two files together.

**Intel PCM support.** The native wrapper is absent, so the PCM radar charts
profiling the six synthetic benchmarks (thesis Fig. 4.13, MOCAST Fig. 3) cannot
be regenerated; the rendered PNGs are in `figures/from_papers/`.

**A patch file for the GCC modification.** The modified `iter.c` is here in
full.

## Artefacts in the bundled traces

Real, and left in the tracked data rather than cleaned out of it — they are part
of what the measurement actually looked like.

- **Negative CpS and CpJ**, from the mutex-free chunk counter's rare torn reads.
  `drop_invalid` removes them.
- **Negative energy readings**, because the original controller differenced the
  32-bit RAPL counters without correcting for wraparound.
  `metrics.EfficiencySampler` corrects it; the bundled traces predate that.
- **CpJ spikes of 10⁷** in one bundled log (the two-phase benchmark under
  Performance), where the energy delta came out near zero.
  `figures/_common.clip_outliers` caps them, as the original analysis did.
- **A factor of two between logged energy and logged CpJ** in the 13-column
  controller-run traces: `cps / energy_j` comes out at about twice the logged
  `cpj`, consistently. Most likely the logged energy is one socket's worth while
  CpJ was computed over both, but the script that wrote these particular files is
  not among those available. Unresolved. It affects the absolute scale of
  `energy_j` in those traces only; the figures use `cps`, `cpj`, `n_cores`,
  `freq_ghz` and the phase bits, which are unaffected.

## Modernization

The original code targets standalone Keras 2.3.1 on TensorFlow 1.x, with
Python 2/3-mixed syntax in places. What changed:

- **`keras` → `tf.keras`** throughout, including the vendored BinaryNet
  `binary_ops` / `binary_layers`.
- **`plotly.plotly` and `cufflinks` → matplotlib.** Both were removed from Plotly
  at version 4.0; the original notebooks cannot run today at any pin. The figure
  scripts are rewrites, not ports.
- **A 19-branch `if/elif` chain of affinity masks → a generated rule.**
- **Ad-hoc `open(..., 'a')` CSV writes scattered through the agent's training
  routine → an optional callback.** The agent does no file I/O.
- **`BinaryConv2D` and the XNOR-Net helpers dropped** — unused here.
- **`rapl.detect_cpu` fixed.** It was a stub that ignored the machine and
  returned Broadwell-EP unconditionally. It now reads `/proc/cpuinfo` and refuses
  unknown models rather than guessing at the DRAM energy unit.

## Citation

This repository reworks:

- **Automatic Energy-Efficiency Monitoring of OpenMP Workloads.** Mirka,
  Sassatelli, Gamatié. ReCoSoC 2019.
- **Online Learning for Dynamic Control of OpenMP Workloads.** Mirka, Sassatelli,
  Gamatié. MOCAST 2020.
- **PhD thesis, Chapter 4**, M. Mirka. *Techniques d'apprentissage pour le
  contrôle adaptatif multi-niveaux du calcul distribué.*

Chapters 5 and 6 of the same thesis are in the sibling repositories `GANNoC`
(RAPIDO 2021) and `m-rwgan` (DATE 2022).

## Licence

Source code — `src/`, `scripts/`, `figures/*.py`, `bench/`, `platform/` — is
licensed under the **Apache License 2.0**; see [`LICENSE`](LICENSE).

Prose and documentation are **CC BY 4.0**. Measured data under `data/` and
`figures/data/` is **CC BY 4.0**.

**Excluded from both grants:**

- `figures/from_papers/**` — the figures as printed in the ReCoSoC 2019 and
  MOCAST 2020 papers and in the thesis. They are the author's own figures, but
  publication transferred copyright, so their reuse is governed by those
  agreements rather than by this licence. They are kept here only for
  side-by-side comparison with the rebuilt figures.

The instrumented `libgomp` the measurements rely on is a modification of GNU
libgomp (GPL-3.0 with the GCC Runtime Library Exception) and is not vendored
here. See [`NOTICE`](NOTICE).
