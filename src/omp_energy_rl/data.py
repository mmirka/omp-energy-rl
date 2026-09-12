"""Loading the SRAD characterization sweep and preparing autoencoder inputs.

The training data is a flat CSV, one row per controller sampling period (500 ms),
written by the original characterization collector as it walked all 209
configurations of the Xeon testbed. It carries no header -- the column meanings
were reconstructed from the collector's ``write()`` call, recorded in
``data/README.md``.

Twenty-eight columns::

    time_s sleep_s chunk joules_total cps cpj n_cores freq_ghz f0 ... f19

``f0`` to ``f19`` are per-core frequency readbacks, one per physical core; they
are loaded for completeness but the autoencoder uses only ``cps``, ``freq_ghz``
and ``n_cores``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Per-core frequency readbacks appended by the collector, one per physical core.
_PER_CORE_FREQ = tuple(f"f{i}" for i in range(20))

#: Column names of the characterization sweep CSV, in order.
CHARACTERIZATION_COLUMNS: Tuple[str, ...] = (
    "time_s",
    "sleep_s",
    "chunk",
    "joules_total",
    "cps",
    "cpj",
    "n_cores",
    "freq_ghz",
) + _PER_CORE_FREQ


def _looks_like_header(fields: Sequence[str]) -> bool:
    try:
        [float(f) for f in fields]
    except ValueError:
        return True
    return False


def load_characterization(path: str | Path) -> pd.DataFrame:
    """Read a characterization sweep CSV into a named-column DataFrame.

    Accepts the headerless original and a re-staged file carrying a header row;
    ``.gz`` is decompressed transparently by pandas.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, nrows=1, dtype=str)
    first_row = list(raw.iloc[0])
    width = len(first_row)
    if width != len(CHARACTERIZATION_COLUMNS):
        raise ValueError(
            f"{path.name} has {width} columns; the characterization schema expects "
            f"{len(CHARACTERIZATION_COLUMNS)} "
            f"({', '.join(CHARACTERIZATION_COLUMNS[:8])}, f0..f19)"
        )
    if _looks_like_header(first_row):
        df = pd.read_csv(path)
        missing = sorted(set(CHARACTERIZATION_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        return df
    return pd.read_csv(path, header=None, names=list(CHARACTERIZATION_COLUMNS))


def drop_invalid(df: pd.DataFrame, columns: Iterable[str] = ("cps", "cpj")) -> pd.DataFrame:
    """Drop rows whose metrics are negative, NaN or infinite.

    Negative CpS/CpJ appear in the recovered traces for two reasons: the
    shared-memory chunk counter was written without a mutex (a rare torn read
    yields nonsense), and the original controller did not correct RAPL counter
    wraparound (yielding negative Joules). Both were filtered downstream at
    analysis time rather than at collection time.
    """
    mask = pd.Series(True, index=df.index)
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            mask &= values.notna() & np.isfinite(values) & (values >= 0)
    return df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Autoencoder input preparation
# --------------------------------------------------------------------------- #

#: The three features the phase autoencoder reconstructs, in order.
PHASE_FEATURES: Tuple[str, str, str] = ("cps", "freq_ghz", "n_cores")

#: Of those, the two that are re-injected at the decoder as "configuration".
CONFIG_FEATURES: Tuple[str, str] = ("freq_ghz", "n_cores")


class FeatureScaler:
    """Per-feature min-max scaling to ``[-1, 1]``.

    The original notebooks fitted one ``sklearn`` ``MinMaxScaler`` per feature and
    carried the three objects around by hand. This bundles them so a fitted
    scaler can be saved next to a trained encoder -- without it, an encoder
    checkpoint is unusable, because the code it emits depends entirely on the
    scaling its inputs were fitted under.
    """

    def __init__(self, features: Sequence[str] = PHASE_FEATURES) -> None:
        self.features: List[str] = list(features)
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        values = df[self.features].to_numpy(dtype=float)
        self.data_min_ = values.min(axis=0)
        self.data_max_ = values.max(axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("FeatureScaler.fit must be called before transform")
        values = df[self.features].to_numpy(dtype=float)
        span = np.where(self.data_max_ - self.data_min_ == 0, 1.0,
                        self.data_max_ - self.data_min_)
        return 2.0 * (values - self.data_min_) / span - 1.0

    def transform_row(self, cps: float, freq_ghz: float, n_cores: float) -> np.ndarray:
        """Scale a single live reading, shaped ``(1, 3)`` for inference."""
        row = pd.DataFrame(
            [[cps, freq_ghz, n_cores]], columns=list(PHASE_FEATURES)
        )
        return self.transform(row[self.features])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            features=np.array(self.features, dtype=object),
            data_min=self.data_min_,
            data_max=self.data_max_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        blob = np.load(Path(path), allow_pickle=True)
        scaler = cls(list(blob["features"]))
        scaler.data_min_ = blob["data_min"]
        scaler.data_max_ = blob["data_max"]
        return scaler


def autoencoder_inputs(
    scaled: np.ndarray,
    features: Sequence[str] = PHASE_FEATURES,
    config_features: Sequence[str] = CONFIG_FEATURES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split scaled features into the autoencoder's two inputs.

    Returns ``(all_features, config_features)``. The second is fed straight to
    the decoder alongside the bottleneck code, which is what forces the
    bottleneck to carry phase rather than configuration.
    """
    index = {name: i for i, name in enumerate(features)}
    config_columns = [index[name] for name in config_features]
    return scaled, scaled[:, config_columns]
