"""Shared setup for the figure scripts: data loading, output directory, styling.

This folder is self-contained on purpose: it imports nothing from ``src/``, and
reads nothing outside ``figures/data/``. Everything the four scripts need --
the CSV schemas, the cleaning helpers, the characterization reduction and the
phase-code bit packing -- lives here, so the chapter's figures can be rebuilt
from this directory alone, with only numpy/pandas/matplotlib installed.

The bundled CSVs are headerless; their schemas are distinguished by column count
and were reconstructed from the writers in the recovered controller sources:

===================  ====  ====================================================
name                 cols  written by
===================  ====  ====================================================
``xeon_sweep``        28   exhaustive (cores x frequency) characterization sweep
``governor_log``      26   the same collector run under a fixed Linux governor
``controller_phase``  13   DQN controller with the phase autoencoder
``controller_plain``   7   DQN controller without it
===================  ====  ====================================================
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # figures are written to files, never shown interactively
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

_PER_CORE_FREQ = [f"f{i}" for i in range(20)]

#: Schema name -> column names, in order.
COLUMNS: Dict[str, Tuple[str, ...]] = {
    "xeon_sweep": tuple(
        ["time_s", "sleep_s", "chunk", "joules_total", "cps", "cpj",
         "n_cores", "freq_ghz"] + _PER_CORE_FREQ
    ),
    "governor_log": tuple(
        # No core count: under a governor the count is not the collector's choice,
        # so it was not logged -- read it off the per-core frequencies.
        ["time_s", "sleep_s", "chunk", "joules_total", "cps", "cpj"] + _PER_CORE_FREQ
    ),
    "controller_phase": (
        "elapsed_s", "sleep_s", "t_act_s", "t_autoencoder_s", "t_dqn_s",
        "energy_j", "cps", "cps_scaled", "cpj", "n_cores", "freq_ghz",
        "phase_bit0", "phase_bit1",
    ),
    "controller_plain": (
        "elapsed_s", "sleep_s", "energy_j", "cps", "cpj", "n_cores", "freq_ghz",
    ),
}

#: Column count -> schema name, for the widths that identify a file on their own.
_BY_WIDTH = {28: "xeon_sweep", 26: "governor_log", 13: "controller_phase"}


def _looks_like_header(fields: Sequence[str]) -> bool:
    try:
        [float(f) for f in fields]
    except ValueError:
        return True
    return False


def load_csv(path, schema: Optional[str] = None) -> pd.DataFrame:
    """Read one bundled CSV into a named-column DataFrame.

    A header row is used when present; otherwise the schema is inferred from the
    column count, which is unambiguous for every width except 7 -- pass
    ``schema="controller_plain"`` for those.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, nrows=1, dtype=str)
    first_row = list(raw.iloc[0])
    width = len(first_row)

    if _looks_like_header(first_row):
        return pd.read_csv(path)
    if schema is None:
        if width not in _BY_WIDTH:
            raise ValueError(
                f"{path.name} has {width} columns and no header; that width is "
                f"ambiguous. Pass schema= explicitly, one of: "
                f"{sorted(n for n, c in COLUMNS.items() if len(c) == width)}"
            )
        schema = _BY_WIDTH[width]
    expected = COLUMNS[schema]
    if len(expected) != width:
        raise ValueError(
            f"{path.name} has {width} columns but schema '{schema}' expects "
            f"{len(expected)}"
        )
    return pd.read_csv(path, header=None, names=list(expected))


def drop_invalid(df: pd.DataFrame, columns: Iterable[str] = ("cps", "cpj")) -> pd.DataFrame:
    """Drop rows whose metrics are negative, NaN or infinite.

    Negative CpS/CpJ appear in the recovered traces because the shared-memory
    chunk counter was written without a mutex (a rare torn read yields nonsense)
    and the original controller did not correct RAPL counter wraparound. Both
    were filtered downstream at analysis time rather than at collection time.
    """
    mask = pd.Series(True, index=df.index)
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            mask &= values.notna() & np.isfinite(values) & (values >= 0)
    return df.loc[mask].reset_index(drop=True)


def clip_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] = ("cps", "cpj"),
    coef: float = 10.0,
) -> pd.DataFrame:
    """Clip values above ``coef`` times the column mean.

    Reproduces the original ``remove_outliners`` helper, which capped rather than
    removed spikes so that row alignment with the rest of the trace survived.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            ceiling = coef * out[column].mean()
            out[column] = out[column].clip(upper=ceiling)
    return out


def characterization_table(
    df: pd.DataFrame,
    drop_first_per_config: int = 1,
) -> pd.DataFrame:
    """Average CpS and CpJ per ``(n_cores, freq_ghz)`` configuration.

    The exhaustive-characterization reduction behind the CpJ surfaces (thesis
    Figs 4.14 and 4.20): run the application in every configuration, take the
    mean over each configuration's samples, and read off the best one.

    ``drop_first_per_config`` discards the leading sample(s) of each
    configuration block, where the metric still reflects the *previous*
    configuration -- the collector applies a configuration and samples one period
    later, so the first row after a switch is a transient.
    """
    if "n_cores" not in df.columns or "freq_ghz" not in df.columns:
        raise ValueError(
            "characterization_table needs 'n_cores' and 'freq_ghz'; this trace has "
            f"{list(df.columns)[:8]}..."
        )
    frame = drop_invalid(df, ("cps", "cpj"))
    if drop_first_per_config > 0:
        block = (
            (frame["n_cores"] != frame["n_cores"].shift())
            | (frame["freq_ghz"] != frame["freq_ghz"].shift())
        ).cumsum()
        frame = frame.loc[
            frame.groupby(block).cumcount() >= drop_first_per_config
        ].reset_index(drop=True)

    return (
        frame.groupby(["n_cores", "freq_ghz"])
        .agg(cps=("cps", "mean"), cpj=("cpj", "mean"), n_samples=("cpj", "size"))
        .reset_index()
        .sort_values(["n_cores", "freq_ghz"])
        .reset_index(drop=True)
    )


def best_configuration(table: pd.DataFrame, metric: str = "cpj") -> pd.Series:
    """The row of a characterization table with the highest ``metric``."""
    return table.loc[table[metric].idxmax()]


def surface(table: pd.DataFrame, metric: str = "cpj"):
    """Reshape a characterization table into ``(cores, freqs, Z)`` grids.

    ``Z`` is indexed ``Z[freq_index, core_index]``, ready for a 3-D surface or a
    heat map. Configurations missing from the table become ``NaN``.
    """
    cores = np.sort(table["n_cores"].unique())
    freqs = np.sort(table["freq_ghz"].unique())
    grid = np.full((len(freqs), len(cores)), np.nan)
    core_index = {c: i for i, c in enumerate(cores)}
    freq_index = {f: i for i, f in enumerate(freqs)}
    for _, row in table.iterrows():
        grid[freq_index[row["freq_ghz"]], core_index[row["n_cores"]]] = row[metric]
    return cores, freqs, grid


def phase_ids(codes) -> np.ndarray:
    """Pack a matrix of +/-1 (or 0/1) bottleneck codes into integer phase ids.

    Big-endian: ``[-1, -1] -> 0``, ``[-1, +1] -> 1``, ``[+1, -1] -> 2``,
    ``[+1, +1] -> 3``. The values are an enumerated type, not an ordering --
    they are arbitrary and differ between training runs (thesis 4.1.3.2).
    """
    bits = (np.asarray(codes) > 0).astype(int)
    if bits.ndim == 1:
        bits = bits.reshape(-1, 1)
    weights = 2 ** np.arange(bits.shape[1] - 1, -1, -1)
    return bits @ weights


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

#: Consistent colours for the four Linux governors used as the baseline.
GOVERNOR_COLORS = {
    "performance": "#c44e52",
    "powersave": "#4c72b0",
    "ondemand": "#55a868",
    "conservative": "#8172b2",
}

#: The controller's own colour, set apart from the governors'.
CONTROLLER_COLOR = "#dd8452"

#: Phase codes are categorical; these are their (arbitrary) colours.
PHASE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig, name: str) -> Path:
    path = OUTPUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path
