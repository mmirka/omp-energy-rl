"""The binarized dense layer used for the autoencoder's bottleneck.

Vendored from the BinaryNet Keras implementation (Courbariaux et al. 2016,
arXiv:1602.02830) and ported from standalone ``keras`` to ``tf.keras``. Only
:class:`BinaryDense` and its weight constraint :class:`Clip` are kept;
``BinaryConv2D`` from the original is dropped as unused.

:class:`BinaryDense` differs from a normal ``Dense`` in one respect: its kernel is
binarized to ``{-H, +H}`` on every forward pass while the *stored* weights stay
real-valued and are clipped to ``[-H, H]``. That is the standard BinaryNet
arrangement -- binary in the forward pass, real-valued for accumulating gradient
updates.

These two classes must be registered as custom objects to load the recovered
``encoder_2bit_adac2_2P.h5`` checkpoint; see
:func:`omp_energy_rl.autoencoder.custom_objects`.
"""

from __future__ import annotations

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers
from tensorflow.keras.layers import Dense, InputSpec

from .binary_ops import binarize


class Clip(constraints.Constraint):
    """Clip weights into ``[min_value, max_value]`` after every update.

    BinaryNet keeps real-valued shadow weights so gradients can accumulate, but
    lets them wander only within the binarization range; beyond it, further
    movement changes nothing in the forward pass and only slows sign flips.
    """

    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class BinaryDense(Dense):
    """A ``Dense`` layer whose kernel is binarized in the forward pass.

    Args:
        units: output width. For the phase bottleneck this is the code width.
        H: binarization magnitude -- the kernel takes values in ``{-H, +H}``.
            ``'Glorot'`` derives it as ``sqrt(1.5 / (fan_in + fan_out))``.
        kernel_lr_multiplier: per-layer learning-rate scale. ``'Glorot'`` derives
            it as the reciprocal of the Glorot ``H``. Recorded on the layer as
            ``lr_multipliers`` for optimizers that consult it; the Adam optimizer
            used throughout this project does not, so in practice it is metadata
            carried for fidelity with the original implementation.
        bias_lr_multiplier: same, for the bias.
    """

    def __init__(
        self,
        units,
        H=1.0,
        kernel_lr_multiplier="Glorot",
        bias_lr_multiplier=None,
        **kwargs,
    ):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.output_dim = units

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(input_shape[-1])

        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
        if self.kernel_lr_multiplier == "Glorot":
            self.kernel_lr_multiplier = np.float32(
                1.0 / np.sqrt(1.5 / (input_dim + self.units))
            )

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "H": self.H,
            "kernel_lr_multiplier": self.kernel_lr_multiplier,
            "bias_lr_multiplier": self.bias_lr_multiplier,
        }
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
