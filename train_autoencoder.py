#!/usr/bin/env python3
"""Train the phase-detection autoencoder on the SRAD characterization sweep.

The autoencoder learns, unsupervised, to reconstruct ``[CpS, frequency, core
count]`` through a discrete bottleneck, with the configuration re-injected at the
decoder. Whatever the bottleneck ends up encoding is what the configuration does
*not* explain -- the workload's execution phase. The number of phases is never
supplied; each distinct code the trained encoder emits is one detected phase, and
the code values are arbitrary (thesis section 4.1.3.2).

Architecture, from thesis Table 4.4 / MOCAST 2020::

    3 -> 100 -> 100 -> 100 -> BinaryDense(code_size) -> BatchNorm -> binary_tanh
      -> concat(frequency, core count) -> 100 -> 100 -> 100 -> 3

MSE, Adam at 1e-5, batch 512, 50 epochs. Runs on CPU in about a minute.

Trains on the bundled SRAD sweep with no hardware::

    python train_autoencoder.py                     # 2-bit code, all 210k rows
    python train_autoencoder.py --code-size 1       # the ReCoSoC 2019 variant
    python train_autoencoder.py --epochs 2 --max-rows 20000   # quick smoke test
    python train_autoencoder.py --plot              # + the phase overlay figure

The encoder and the fitted scaler are saved together, and must stay together: a
code is only meaningful relative to the scaling its inputs were fitted under.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS = REPO_ROOT / "results"

import tensorflow as tf  # noqa: E402

from omp_energy_rl import data  # noqa: E402
from omp_energy_rl.autoencoder import (  # noqa: E402
    AutoencoderConfig,
    build_phase_autoencoder,
    phase_ids,
    train_phase_autoencoder,
)

#: The SRAD sweep: 19 core counts x 11 frequencies x 1000 samples at 0.5 s.
DEFAULT_TRACE = REPO_ROOT / "data" / "srad_allConf_19c11f.csv.gz"


def plot_phase_overlay(frame, ids, path, window=400, offset=None):
    """CpS over time, coloured by the phase the encoder assigned to each sample.

    The window defaults to the middle of the sweep. That matters: the file is
    ordered by configuration, one 1000-sample block each, so a window near the
    start sits in the 1-core blocks where CpS never leaves the bottom of the
    global range and every sample lands in the same code. The midpoint is a
    representative block, not a flattering one -- override with ``offset`` to
    look anywhere else.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if offset is None:
        offset = max(0, len(ids) // 2 - window // 2)
    start = min(offset, max(0, len(ids) - window))
    stop = min(start + window, len(ids))
    time_s = frame["time_s"].to_numpy()[start:stop]
    cps = frame["cps"].to_numpy()[start:stop]
    codes = ids[start:stop]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, cps, color="0.75", linewidth=0.8, zorder=1)
    palette = plt.get_cmap("tab10")
    for value in np.unique(codes):
        mask = codes == value
        ax.scatter(
            time_s[mask], cps[mask], s=12, zorder=2,
            color=palette(int(value) % 10), label=f"phase {int(value)}",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("CpS (chunks/s)")
    ax.set_title(
        f"CpS coloured by detected phase (samples {start}-{stop}: "
        f"{int(frame['n_cores'].to_numpy()[start])} cores @ "
        f"{frame['freq_ghz'].to_numpy()[start]:.1f} GHz)"
    )
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trace",
        default=str(DEFAULT_TRACE),
        help="characterization sweep CSV, 28 columns (default: the bundled SRAD sweep)",
    )
    parser.add_argument(
        "--code-size",
        type=int,
        default=2,
        help="bottleneck width in bits; 2**code-size representable phases "
        "(default: 2, as in thesis Table 4.4; use 1 for the ReCoSoC 2019 variant)",
    )
    parser.add_argument("--epochs", type=int, default=AutoencoderConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=AutoencoderConfig.batch_size)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="tail fraction of the trace held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap the number of rows used, for a quick run",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also write phase_overlay.png -- CpS coloured by detected phase",
    )
    parser.add_argument(
        "--plot-offset",
        type=int,
        default=None,
        help="first sample of the --plot window (default: the middle of the sweep)",
    )
    parser.add_argument(
        "--plot-window", type=int, default=400, help="samples to plot (default: 400)"
    )
    parser.add_argument("--out-dir", default=str(RESULTS / "autoencoder"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # Both RNGs: TensorFlow's drives weight initialisation and batch shuffling,
    # so seeding NumPy alone does not make a run reproducible.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    frame = data.load_characterization(args.trace)
    frame = data.drop_invalid(frame, ("cps", "cpj"))
    if args.max_rows:
        frame = frame.iloc[: args.max_rows]
    print(f"{len(frame)} valid samples from {args.trace}")
    print(
        f"  {frame['n_cores'].nunique()} core counts x "
        f"{frame['freq_ghz'].nunique()} frequencies = "
        f"{len(frame.groupby(['n_cores', 'freq_ghz']))} configurations observed"
    )

    scaler = data.FeatureScaler().fit(frame)
    scaled = scaler.transform(frame)
    all_features, config_features = data.autoencoder_inputs(scaled)

    split = int(len(scaled) * (1.0 - args.validation_split))
    train = (all_features[:split], config_features[:split])
    validation = (all_features[split:], config_features[split:])
    print(f"  {split} train / {len(scaled) - split} validation rows")

    config = AutoencoderConfig(
        code_size=args.code_size, epochs=args.epochs, batch_size=args.batch_size
    )
    autoencoder, encoder = build_phase_autoencoder(config)
    print(
        f"\nautoencoder: 3 -> {config.hidden_units}x3 -> BinaryDense({config.code_size})"
        f" -> concat(config) -> {config.hidden_units}x3 -> 3  "
        f"({autoencoder.count_params()} params)"
    )

    started = time.time()
    history = train_phase_autoencoder(
        autoencoder, train[0], train[1], validation=validation, config=config
    )
    elapsed = time.time() - started
    print(
        f"\ntrained {config.epochs} epochs in {elapsed:.1f}s "
        f"({elapsed / config.epochs:.2f}s/epoch); "
        f"final loss {history.history['loss'][-1]:.5f}, "
        f"val_loss {history.history['val_loss'][-1]:.5f}"
    )

    codes = encoder.predict(all_features)
    ids = phase_ids(codes)
    unique, counts = np.unique(ids, return_counts=True)
    print(f"\ndetected {len(unique)} distinct phase code(s):")
    for value, count in zip(unique, counts):
        share = 100.0 * count / len(ids)
        mean_cps = frame["cps"].to_numpy()[ids == value].mean()
        mean_cpj = frame["cpj"].to_numpy()[ids == value].mean()
        print(
            f"  code {value}: {count:>7} samples ({share:5.1f}%)  "
            f"mean CpS {mean_cps:12.1f}  mean CpJ {mean_cpj:10.2f}"
        )
    if len(unique) == 1:
        print(
            "  NOTE: a single code means no phase structure was separated. Either the "
            "workload really is single-phase, or training has not converged -- try "
            "more epochs."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / f"encoder_{args.code_size}bit.h5"
    scaler_path = out_dir / f"encoder_{args.code_size}bit_scaler.npz"
    encoder.save(str(encoder_path))
    scaler.save(scaler_path)
    print(f"\nwrote {encoder_path}\nwrote {scaler_path}")
    print("Both are needed at inference time -- a code is meaningless without its scaler.")

    if args.plot:
        plot_path = out_dir / "phase_overlay.png"
        plot_phase_overlay(
            frame, ids, plot_path, window=args.plot_window, offset=args.plot_offset
        )
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
