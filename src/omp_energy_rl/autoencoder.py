"""The phase-detection autoencoder.

The controller needs to know *which execution phase* an application is currently
in -- compute-bound, memory-bound, something in between -- without being told in
advance how many phases there are, and without profiling the application first.
This is the unsupervised model that supplies it.

The architecture (thesis Table 4.4, Fig. 4.8) is an autoencoder over three
features -- ``[CpS, frequency, core count]``, each min-max scaled to ``[-1, 1]``
-- with two deliberate design choices:

1. **A discrete bottleneck.** The code layer is a :class:`~omp_energy_rl.
   binary_layers.BinaryDense` followed by batch normalization and a
   ``binary_tanh`` activation, so the code is a vector of ``+-1``. A code of
   width *n* can name ``2**n`` phases. The thesis uses ``code_size=2``; the
   earlier ReCoSoC 2019 version used ``code_size=1``.
2. **The configuration is re-injected at the decoder.** The decoder receives
   ``concat(code, [frequency, core count])``, so it already knows the
   configuration and the bottleneck gains nothing by encoding it. What is left
   for the code to carry is the part of CpS that the configuration does *not*
   explain -- which is the workload's phase.

The result is an unsupervised classifier: each distinct code value the trained
encoder emits corresponds to one detected phase. **The code values themselves are
arbitrary** and vary between training runs -- treat them as an enumerated type,
not as an ordering (thesis section 4.1.3.2).

Training takes 50 epochs at well under 5 s/epoch, so it is done offline; a single
phase prediction takes under 5 ms, which is what makes online use possible inside
the controller's 500 ms sampling period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Input,
    concatenate,
)
from tensorflow.keras.optimizers import Adam

from .binary_layers import BinaryDense, Clip
from .binary_ops import binary_tanh

#: Name of the full-feature input tensor, kept for checkpoint compatibility.
INPUT_ALL = "DATA_All"

#: Name of the configuration input re-injected at the decoder.
INPUT_CONF = "DATA_Conf"


@dataclass
class AutoencoderConfig:
    """Hyperparameters of the phase autoencoder (thesis Table 4.4)."""

    input_size: int = 3
    """``[CpS, frequency, core count]``."""

    code_size: int = 2
    """Bottleneck width in bits: ``2**code_size`` representable phases.
    2 for thesis Ch.4 / MOCAST 2020, 1 for ReCoSoC 2019."""

    config_size: int = 2
    """``[frequency, core count]`` -- the part re-injected at the decoder."""

    hidden_units: int = 100
    """Width of each of the three encoder and three decoder hidden layers."""

    learning_rate: float = 1e-5
    """Adam learning rate. Deliberately small: the straight-through binarizer
    makes the bottleneck's effective loss surface step-like, and larger rates
    make the code thrash between values instead of settling."""

    batch_size: int = 512
    epochs: int = 50

    bn_epsilon: float = 1e-6
    bn_momentum: float = 0.9

    use_bias: bool = False
    """The bottleneck's ``BinaryDense`` carries no bias in the original design."""

    binarization_scale: Any = "Glorot"
    """``H`` for :class:`BinaryDense` -- ``'Glorot'`` derives it from the fan
    in/out."""


def build_phase_autoencoder(
    config: Optional[AutoencoderConfig] = None,
) -> Tuple[Model, Model]:
    """Build the autoencoder and the encoder that shares its weights.

    Returns ``(autoencoder, encoder)``. Train the first; use the second at
    inference time -- it maps scaled ``[CpS, frequency, core count]`` straight to
    the ``+-1`` phase code, skipping the decoder entirely.

    The returned autoencoder is already compiled with Adam and MSE.
    """
    cfg = config or AutoencoderConfig()

    input_all = Input(shape=(cfg.input_size,), name=INPUT_ALL)
    input_conf = Input(shape=(cfg.config_size,), name=INPUT_CONF)

    # Encoder: three linear layers down to the discrete bottleneck.
    x = Dense(cfg.hidden_units, activation="linear")(input_all)
    x = Dense(cfg.hidden_units, activation="linear")(x)
    x = Dense(cfg.hidden_units, activation="linear")(x)
    x = BinaryDense(
        cfg.code_size,
        H=cfg.binarization_scale,
        kernel_lr_multiplier=cfg.binarization_scale,
        use_bias=cfg.use_bias,
    )(x)
    x = BatchNormalization(epsilon=cfg.bn_epsilon, momentum=cfg.bn_momentum)(x)
    code = Activation(binary_tanh, name="phase_code")(x)

    # Decoder: the code, plus the configuration it is not required to encode.
    y = concatenate([code, input_conf])
    y = Dense(cfg.hidden_units, activation="linear")(y)
    y = Dense(cfg.hidden_units, activation="linear")(y)
    y = Dense(cfg.hidden_units, activation="linear")(y)
    output = Dense(cfg.input_size, activation="linear")(y)

    autoencoder = Model(inputs=[input_all, input_conf], outputs=output)
    autoencoder.compile(optimizer=Adam(learning_rate=cfg.learning_rate), loss="mse")

    encoder = Model(inputs=input_all, outputs=code, name="phase_encoder")
    return autoencoder, encoder


def train_phase_autoencoder(
    autoencoder: Model,
    train_all: np.ndarray,
    train_conf: np.ndarray,
    validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    config: Optional[AutoencoderConfig] = None,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Fit the autoencoder to reconstruct its own ``train_all`` input.

    ``validation``, if given, is a ``(all, conf)`` pair held out the same way.
    """
    cfg = config or AutoencoderConfig()
    validation_data = None
    if validation is not None:
        val_all, val_conf = validation
        validation_data = ([val_all, val_conf], val_all)
    return autoencoder.fit(
        {INPUT_ALL: train_all, INPUT_CONF: train_conf},
        train_all,
        validation_data=validation_data,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        verbose=verbose,
    )


def custom_objects() -> Dict[str, Any]:
    """Custom objects needed to deserialize a saved encoder or autoencoder.

    ``BinaryDense`` and ``Clip`` are the vendored BinaryNet pieces; ``binary_tanh``
    is the bottleneck activation, which Keras serializes by name only.
    """
    return {"BinaryDense": BinaryDense, "Clip": Clip, "binary_tanh": binary_tanh}


def load_pretrained_encoder(path: str | Path) -> Model:
    """Load a saved phase encoder, wiring up the BinaryNet custom objects.

    Loads the recovered ``encoder_2bit_adac2_2P.h5`` -- the 2-bit encoder trained
    on the two-phase synthetic benchmark and used for the thesis Ch.4 multi-phase
    control results.

    Note what a checkpoint does *not* carry: the min-max scaling its inputs were
    fitted under. An encoder is only meaningful together with the
    :class:`~omp_energy_rl.data.FeatureScaler` fitted on the same trace, and the
    scaler for this particular checkpoint did not survive (see
    the README's 'What survived, and what did not' section).
    """
    return tf.keras.models.load_model(
        str(path), custom_objects=custom_objects(), compile=False
    )


def phase_ids(codes: np.ndarray) -> np.ndarray:
    """Map ``+-1`` code vectors to small non-negative integer phase ids.

    ``[-1, -1] -> 0``, ``[-1, +1] -> 1``, ``[+1, -1] -> 2``, ``[+1, +1] -> 3``.
    Purely a convenience for plotting and grouping; the ids carry no order, for
    the same reason the raw codes do not.
    """
    bits = (np.asarray(codes) > 0).astype(int)
    if bits.ndim == 1:
        bits = bits.reshape(-1, 1)
    weights = 2 ** np.arange(bits.shape[1] - 1, -1, -1)
    return bits @ weights
