#!/usr/bin/env python3
"""CpS with the detected execution phase overlaid -- thesis Fig. 4.11.

The validation figure for phase detection: plot CpS over time and colour each
sample by the phase the autoencoder assigned it. If the model works, the colours
line up with the visible structure in the CpS trace -- and they do so having
never been told how many phases there are, or which sample belongs to which.

The phase codes drawn here were emitted **live**, by the autoencoder running
inside the controller during the bundled run. Nothing is recomputed and no model
is loaded, which is what keeps this folder free of TensorFlow. For the same
picture from a *freshly trained* encoder, use ``train_autoencoder.py --plot`` at
the repository root.

::

    cd figures && python phase_overlay.py
"""

import argparse
from pathlib import Path

import numpy as np

from _common import DATA, PHASE_COLORS, apply_style, load_csv, phase_ids, save
import matplotlib.pyplot as plt

DEFAULT_RUN = DATA / "controller_runs" / "history_srad_ae2bit__27_07_2021_13_22_59.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--window", type=int, default=400,
        help="samples to show; SRAD alternates phases every few samples, so a "
        "short window is what makes the correspondence readable (default: 400)"
    )
    parser.add_argument(
        "--offset", type=int, default=3000, help="first sample to show (default: 3000)"
    )
    parser.add_argument("--out", default="phase_overlay.png")
    args = parser.parse_args()

    apply_style()

    frame = load_csv(args.run, schema="controller_phase")
    ids = phase_ids(frame[["phase_bit0", "phase_bit1"]].to_numpy())
    source = (
        "phase codes emitted live by the autoencoder during "
        f"{Path(args.run).name}"
    )

    window = slice(args.offset, args.offset + args.window)
    frame = frame.iloc[window].reset_index(drop=True)
    ids = ids[window]
    time_s = frame["elapsed_s"].to_numpy()
    cps = frame["cps"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    axes[0].plot(time_s, cps, lw=0.6, color="#666666", zorder=1)
    for index, phase in enumerate(sorted(np.unique(ids))):
        mask = ids == phase
        axes[0].scatter(
            time_s[mask], cps[mask], s=5, zorder=2,
            color=PHASE_COLORS[index % len(PHASE_COLORS)],
            label=f"phase {phase} (n={mask.sum()})",
        )
    axes[0].set_ylabel("CpS")
    axes[0].set_title(
        "Execution phases detected on the CpS profile\n" + source, fontsize=10
    )
    axes[0].legend(markerscale=3, fontsize=8, ncol=len(np.unique(ids)))

    axes[1].step(time_s, ids, where="post", lw=0.8, color="#8172b2")
    axes[1].set_yticks(sorted(np.unique(ids)))
    axes[1].set_ylabel("phase code")
    axes[1].set_xlabel("time (s)")

    save(fig, args.out)

    print(f"\n{len(np.unique(ids))} phase(s) over {len(ids)} samples shown:")
    for phase in sorted(np.unique(ids)):
        mask = ids == phase
        print(
            f"  phase {phase}: {mask.sum():>5} samples, "
            f"mean CpS {cps[mask].mean():12.1f}"
        )
    print(
        "\nPhase code values are arbitrary -- an enumerated type, not an ordering "
        "(thesis 4.1.3.2)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
