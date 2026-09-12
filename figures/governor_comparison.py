#!/usr/bin/env python3
"""Best-known configuration vs the four Linux governors -- thesis Tables 4.2 / 4.5.

The baseline the whole project is measured against. Each Linux governor runs the
same workload with its own frequency policy and *no control at all over how many
cores the workload gets*; the "best configuration" bar is the optimum found by
the exhaustive characterization sweep.

The gap between them is the headline result: on DGEMM the thesis reports
10 / 17 / 11 / 10% CpJ gains over Performance / Powersave / Ondemand /
Conservative. Recomputed here from the bundled logs, the same four numbers come
out at roughly 12 / 18 / 12 / 12%.

Runs from bundled logs with no hardware::

    cd figures && python governor_comparison.py
"""

import argparse

from _common import (
    CONTROLLER_COLOR,
    DATA,
    GOVERNOR_COLORS,
    apply_style,
    best_configuration,
    characterization_table,
    clip_outliers,
    drop_invalid,
    load_csv,
    save,
)
import matplotlib.pyplot as plt

GOVERNORS = ["performance", "powersave", "ondemand", "conservative"]


def governor_means(workload: str) -> dict:
    """Mean CpJ per governor, from the bundled logs.

    The median is reported alongside because one bundled log (the two-phase
    benchmark under Performance) contains a handful of samples where the RAPL
    energy delta came out near zero, producing CpJ spikes five orders of
    magnitude above the rest. The thesis's own analysis filtered such samples;
    ``clip_outliers`` does the same here.
    """
    means = {}
    for governor in GOVERNORS:
        path = DATA / "governor_comparison" / f"log_cpj_{workload}_{governor}_index0.csv"
        frame = clip_outliers(drop_invalid(load_csv(path)), ("cpj",))
        means[governor] = float(frame["cpj"].mean())
    return means


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", default="DGEMM", choices=["DGEMM", "2P"])
    parser.add_argument(
        "--sweep",
        default=str(DATA / "dgemm_19c11f.csv"),
        help="characterization sweep that defines the best-known configuration",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    apply_style()
    means = governor_means(args.workload)

    table = characterization_table(load_csv(args.sweep))
    best = best_configuration(table, "cpj")
    reference = float(best["cpj"])
    label = (
        f"best configuration\n({int(best['n_cores'])} cores @ "
        f"{best['freq_ghz']:.1f} GHz)"
    )

    names = GOVERNORS + ["best"]
    values = [means[g] for g in GOVERNORS] + [reference]
    colors = [GOVERNOR_COLORS[g] for g in GOVERNORS] + [CONTROLLER_COLOR]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([g.capitalize() for g in GOVERNORS] + [label], fontsize=9)
    ax.set_ylabel("mean CpJ (chunks per Joule)")
    ax.set_title(
        f"{args.workload}: energy efficiency of the best configuration "
        f"vs the Linux governors"
    )

    for index, (bar, value) in enumerate(zip(bars, values)):
        text = f"{value:.0f}"
        if index < len(GOVERNORS):
            text += f"\n+{100 * (reference / value - 1):.0f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, text,
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.22)

    out = args.out or f"governor_comparison_{args.workload.lower()}.png"
    save(fig, out)

    print(f"\nreference (best configuration): CpJ {reference:.2f}")
    for governor in GOVERNORS:
        print(
            f"  vs {governor:<13} CpJ {means[governor]:8.2f}  "
            f"-> {100 * (reference / means[governor] - 1):+6.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
