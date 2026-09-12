# Figures

Four scripts that rebuild the chapter's figures **from the data in this folder
only** — no hardware, no training, no network access, and nothing imported from
`../src/`. Everything they need lives in `_common.py` and `data/`, so this
directory is self-contained: copy it anywhere with numpy, pandas and matplotlib
installed and the figures rebuild.

```bash
conda activate omp-energy-rl
cd figures
python cpj_surface.py
python governor_comparison.py
python controller_trace.py
python phase_overlay.py
```

Output lands in `output/` (git-ignored). The originals, as printed in the papers
and the thesis, are in [`from_papers/`](from_papers/) for side-by-side
comparison. The inputs are in [`data/`](data/):

```
data/dgemm_19c11f.csv                 the DGEMM characterization sweep (28 cols)
data/governor_comparison/*.csv        4 governors x 2 workloads (26 cols)
data/controller_runs/*.csv            3 recovered controller runs (13 or 7 cols)
```

All of them are headerless; `_common.load_csv` picks the schema by column count.

The original notebooks that produced these used `plotly.plotly` and `cufflinks`,
both removed from Plotly at version 4.0 — they cannot run today at any pin. These
are matplotlib rewrites, not ports.

| Script | Rebuilds | Reads |
|---|---|---|
| `cpj_surface.py` | Thesis Figs. 4.14, 4.20 — CpJ over all 209 configurations | the DGEMM characterization sweep |
| `governor_comparison.py` | Thesis Tables 4.2, 4.5 — best configuration vs the four Linux governors | the governor comparison logs |
| `controller_trace.py` | Thesis Figs. 4.18–4.24 — CpJ and the chosen configuration over a controlled run | the bundled controller runs |
| `phase_overlay.py` | Thesis Fig. 4.11 — CpS coloured by detected phase | a controller run's live phase codes |

## `cpj_surface.py`

The exhaustive characterization of one workload as a 3-D surface and a heat map,
with the best configuration marked. Reducing the bundled DGEMM sweep gives
**19 cores @ 2.1 GHz, mean CpJ 166.2**; the thesis reports 18 cores @ 2.1 GHz,
mean CpJ 166 (section 4.3.1) — the frequency and the efficiency value reproduce
exactly, the core count lands one apart.

The core-count axis is the one the Linux governors cannot touch.

## `governor_comparison.py`

```bash
python governor_comparison.py                  # DGEMM
python governor_comparison.py --workload 2P    # the two-phase benchmark
```

Recomputed from the bundled logs, the DGEMM gains come out as:

| vs governor | thesis §4.3.1 | recomputed here |
|---|---|---|
| Performance | 10% | 11.7% |
| Powersave | 17% | 18.4% |
| Ondemand | 11% | 12.3% |
| Conservative | 10% | 12.1% |

One bundled log — the two-phase benchmark under Performance — contains a few
samples where the RAPL energy delta came out near zero, giving CpJ spikes five
orders of magnitude above everything else. `_common.clip_outliers` caps them
rather than dropping them, which is what the thesis's own analysis did.

## `controller_trace.py`

```bash
python controller_trace.py                   # SRAD, phase-aware
python controller_trace.py --run dgemm       # single-phase, phase-free
python controller_trace.py --run two_phase   # two-phase, phase-free (the ablation)
```

CpJ, the chosen configuration, and (where recorded) the phase code, against a
shared time axis, with the end of the exploration period marked.

A run is 5000–15000 samples at 500 ms; drawn raw it is a solid block, so a
rolling mean is drawn over the raw trace rather than instead of it — per-period
CpJ really is that noisy, and smoothing it away would misrepresent the signal the
agent learns from.

**Read the exploration line carefully.** On the SRAD run, exploitation begins at
1024 s but mean CpJ keeps climbing until roughly 2000 s. That is not a
discrepancy: experience replay holds 2048 transitions, so it takes another full
buffer's worth of *greedy* experience to flush the random actions out and refit
the Q-network on what the greedy policy actually sees. Exploration ending is when
learning stops being random, not when it finishes.

## `phase_overlay.py`

```bash
python phase_overlay.py
```

The validation figure for phase detection. It plots the codes the autoencoder
emitted **live** during a bundled SRAD run — nothing is recomputed and no model
is loaded, which is what keeps this folder free of TensorFlow. For the same
picture from an encoder you just trained, use `python train_autoencoder.py
--plot` at the repository root. In the default 400-sample window, one
code sits on the high-CpS peaks and the other on the baseline: the model
recovered SRAD's two-phase structure having never been told there were two
phases, or which sample belonged to which.

Phase code values are arbitrary and change between training runs — an enumerated
type, not an ordering.
