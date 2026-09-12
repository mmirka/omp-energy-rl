#!/usr/bin/env python3
"""CpJ across all 209 configurations -- thesis Figs. 4.14 and 4.20.

The exhaustive characterization of one workload: mean CpJ for every combination
of core count and frequency. This is the picture the whole project is about.
Reading it:

* the **core-count axis** is the one the Linux governors cannot touch. They tune
  frequency only, so at best they can find the best point on one column of this
  surface -- and for a memory-bound workload the best column is nowhere near the
  one they are standing in.
* the surface has **a single broad optimum**, not a spiky landscape, which is why
  an agent can find a near-best configuration from 2048 samples over a
  209-point space.

Reproduces from the bundled DGEMM sweep with no hardware and no training::

    cd figures && python cpj_surface.py
"""

import argparse

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from _common import (
    DATA,
    apply_style,
    best_configuration,
    characterization_table,
    load_csv,
    save,
    surface,
)
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trace", default=str(DATA / "dgemm_19c11f.csv")
    )
    parser.add_argument("--schema", default=None)
    parser.add_argument("--label", default="DGEMM")
    parser.add_argument("--out", default="cpj_surface.png")
    args = parser.parse_args()

    apply_style()
    table = characterization_table(load_csv(args.trace, schema=args.schema))
    cores, freqs, grid = surface(table, "cpj")
    best = best_configuration(table, "cpj")

    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    mesh_c, mesh_f = np.meshgrid(cores, freqs)
    ax3d.plot_surface(
        mesh_c, mesh_f, grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95
    )
    ax3d.set_xlabel("cores")
    ax3d.set_ylabel("frequency (GHz)")
    ax3d.set_zlabel("CpJ")
    ax3d.set_title(f"{args.label}: energy efficiency over the action space")
    ax3d.view_init(elev=26, azim=-128)

    ax = fig.add_subplot(1, 2, 2)
    ax.grid(False)
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(cores[0] - 0.5, cores[-1] + 0.5, freqs[0] - 0.05, freqs[-1] + 0.05),
    )
    ax.scatter(
        best["n_cores"], best["freq_ghz"],
        marker="*", s=300, color="white", edgecolor="black", linewidth=0.8, zorder=3,
    )
    ax.annotate(
        f"best: {int(best['n_cores'])} cores @ {best['freq_ghz']:.1f} GHz\n"
        f"CpJ {best['cpj']:.0f}",
        xy=(best["n_cores"], best["freq_ghz"]),
        xytext=(-12, -30), textcoords="offset points",
        ha="right", color="white", fontsize=9,
    )
    ax.set_xlabel("cores")
    ax.set_ylabel("frequency (GHz)")
    ax.set_title(f"{len(table)} configurations, mean CpJ")
    # The global grid style would otherwise be applied to (and warned about on)
    # the colorbar's own axes.
    with plt.rc_context({"axes.grid": False}):
        fig.colorbar(image, ax=ax, label="CpJ")

    save(fig, args.out)
    print(
        f"best configuration: {int(best['n_cores'])} cores @ {best['freq_ghz']:.1f} GHz "
        f"-> CpJ {best['cpj']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
