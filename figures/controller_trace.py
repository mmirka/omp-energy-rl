#!/usr/bin/env python3
"""What the online controller actually did -- thesis Figs. 4.18-4.24.

Three stacked panels against a shared time axis:

* **CpJ** -- the reward signal, and the thing being optimized.
* **The action** -- core count and frequency, i.e. what the controller chose.
* **The phase code** -- present only for runs that used the autoencoder.

The vertical line marks the end of the exploration period (2048 steps = 1024 s).
To the left of it the controller chooses at random; to the right it follows its
Q-network's argmax.

The improvement does not appear at that line, and should not be expected to. In
the bundled SRAD run, exploitation starts at 1024 s but mean CpJ keeps climbing
until roughly 2000 s, and the mean core count with it -- because the replay
buffer holds only 2048 transitions, so it takes about another full buffer's worth
of *exploitation* experience to flush out the random actions and refit the
Q-network on what the greedy policy is actually seeing. Exploration ending is
when learning stops being random, not when it finishes.

Runs from bundled logs, no hardware and no training::

    cd figures && python controller_trace.py                 # SRAD, phase-aware
    cd figures && python controller_trace.py --run dgemm     # DGEMM, phase-free
    cd figures && python controller_trace.py --run two_phase # 2-phase, ablation
"""

import argparse

import numpy as np

from _common import (
    CONTROLLER_COLOR,
    DATA,
    PHASE_COLORS,
    apply_style,
    clip_outliers,
    drop_invalid,
    load_csv,
    phase_ids,
    save,
)
import matplotlib.pyplot as plt

RUNS = {
    "srad": (
        DATA / "controller_runs" / "history_srad_ae2bit__27_07_2021_13_22_59.csv",
        "controller_phase",
        "SRAD (Rodinia), phase-aware controller",
    ),
    "dgemm": (
        DATA / "controller_runs" / "history_reference_cpj_03-08-2021_19-36-05.csv",
        "controller_plain",
        "single-phase synthetic workload, phase-free controller",
    ),
    "two_phase": (
        DATA / "controller_runs" / "history_reference_cpj_06_08_2021_11_44_16.csv",
        "controller_plain",
        "two-phase workload, phase-free controller (the ablation of Fig. 4.24)",
    ),
}

#: 2048 steps at the 500 ms sampling period.
EXPLORATION_SECONDS = 1024.0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="srad", help=f"one of {sorted(RUNS)}, or a path")
    parser.add_argument("--schema", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--smooth", type=int, default=120,
        help="rolling-mean window in samples for the readable overlay "
        "(120 samples = 60 s at the 500 ms period; 0 disables)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.run in RUNS:
        path, schema, title = RUNS[args.run]
        name = args.run
    else:
        path, schema, title = args.run, args.schema, args.run
        name = "run"
    title = args.title or title

    apply_style()
    frame = clip_outliers(
        drop_invalid(load_csv(path, schema=schema)), ("cpj",)
    )
    time_s = frame["elapsed_s"].to_numpy()
    has_phase = "phase_bit0" in frame.columns

    fig, axes = plt.subplots(
        3 if has_phase else 2, 1, figsize=(10, 7 if has_phase else 5.5), sharex=True
    )

    # A run is 5000-15000 samples at 500 ms; drawn raw, the per-sample trace is a
    # solid block. Raw stays as the light background -- the scatter is real, and
    # hiding it would misrepresent how noisy per-period CpJ is -- with a rolling
    # mean over it for the trend.
    def smoothed(series):
        if args.smooth <= 1:
            return None
        return series.rolling(args.smooth, min_periods=1, center=True).mean()

    axes[0].plot(time_s, frame["cpj"], lw=0.4, color=CONTROLLER_COLOR, alpha=0.30)
    trend = smoothed(frame["cpj"])
    if trend is not None:
        axes[0].plot(time_s, trend, lw=1.6, color=CONTROLLER_COLOR)
    axes[0].set_ylabel("CpJ")
    axes[0].set_title(
        f"{title}\n{len(frame)} control steps"
        + (f"; bold = {args.smooth}-sample rolling mean" if trend is not None else "")
    )

    axes[1].plot(time_s, frame["n_cores"], lw=0.4, color="#4c72b0", alpha=0.25)
    cores_trend = smoothed(frame["n_cores"])
    if cores_trend is not None:
        axes[1].plot(time_s, cores_trend, lw=1.6, color="#4c72b0")
    axes[1].set_ylabel("cores", color="#4c72b0")
    axes[1].tick_params(axis="y", labelcolor="#4c72b0")
    axes[1].set_ylim(0, 20)
    twin = axes[1].twinx()
    twin.plot(time_s, frame["freq_ghz"], lw=0.4, color="#55a868", alpha=0.25)
    freq_trend = smoothed(frame["freq_ghz"])
    if freq_trend is not None:
        twin.plot(time_s, freq_trend, lw=1.6, color="#55a868")
    twin.set_ylabel("frequency (GHz)", color="#55a868")
    twin.tick_params(axis="y", labelcolor="#55a868")
    twin.set_ylim(1.15, 2.25)
    twin.grid(False)

    if has_phase:
        codes = frame[["phase_bit0", "phase_bit1"]].to_numpy()
        ids = phase_ids(codes)
        # Phase codes are categorical, so a rolling mean would be meaningless.
        # Shown instead as one horizontal band per code -- which reads as
        # "which phase, when" without pretending the values are ordered.
        for index, phase in enumerate(sorted(np.unique(ids))):
            mask = ids == phase
            axes[2].scatter(
                time_s[mask], np.full(mask.sum(), phase), s=1.5,
                color=PHASE_COLORS[index % len(PHASE_COLORS)],
            )
        axes[2].set_ylabel("phase code")
        axes[2].set_yticks(sorted(np.unique(ids)))
        axes[2].set_ylim(-0.5, max(ids) + 0.5)

    for axis in axes:
        axis.axvline(EXPLORATION_SECONDS, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0].annotate(
        "exploration ends\n(2048 steps)",
        xy=(EXPLORATION_SECONDS, axes[0].get_ylim()[1]),
        xytext=(8, -14), textcoords="offset points", fontsize=8, va="top",
    )
    axes[-1].set_xlabel("time (s)")

    save(fig, args.out or f"controller_trace_{name}.png")

    explore = frame[frame["elapsed_s"] <= EXPLORATION_SECONDS]
    exploit = frame[frame["elapsed_s"] > EXPLORATION_SECONDS]
    if len(exploit):
        print(f"\nmean CpJ during exploration : {explore['cpj'].mean():9.2f}")
        print(f"mean CpJ after exploration  : {exploit['cpj'].mean():9.2f}")
        print(
            f"settled configuration       : {exploit['n_cores'].mean():.2f} cores @ "
            f"{exploit['freq_ghz'].mean():.2f} GHz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
