"""Straight-through binarization ops for the autoencoder's discrete bottleneck.

Vendored from the BinaryNet Keras implementation and ported from standalone
``keras`` to ``tf.keras``. Only the pieces the phase autoencoder actually uses are
kept -- the weight/activation binarization primitives; the XNOR-Net helpers and
the binarized convolution of the original are dropped as unused here.

Reference:
    Courbariaux, Hubara, Soudry, El-Yaniv, Bengio.
    *BinaryNet: Training Deep Neural Networks with Weights and Activations
    Constrained to +1 or -1.* arXiv:1602.02830, 2016.

Why this matters for phase detection: the autoencoder's bottleneck must emit a
*discrete* code, so that each distinct code value can be read as "this is
execution phase N". A plain continuous bottleneck would give a smooth latent that
no downstream controller could treat as a phase id. Binarizing has zero gradient
almost everywhere, so the ``round_through`` straight-through estimator passes the
identity gradient backwards while the forward pass stays hard.
"""

from __future__ import annotations

from tensorflow.keras import backend as K


def round_through(x):
    """Round to the nearest integer, but propagate gradients as if identity.

    The straight-through estimator: forward is ``round(x)``, backward is
    ``d/dx x``. Credit for the trick to Sergey Ioffe
    (http://stackoverflow.com/a/36480182).
    """
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    """Hard sigmoid, clipped to ``[0, 1]``.

    Deliberately *not* ``K.hard_sigmoid``: BinaryNet uses ``0.5x + 0.5`` clipped,
    which is a different slope from Keras' conventional definition.
    """
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    """Binarize to ``{0, 1}`` with a straight-through gradient."""
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    """Binarize to ``{-1, +1}`` with a straight-through gradient.

    Behaves like ``sign`` on the forward pass, and like ``hard_tanh`` on the
    backward pass -- so the gradient vanishes once ``|x| > 1``, which is what
    keeps the pre-activation from drifting arbitrarily far from the threshold.

    This is the bottleneck activation: with a code of width *n*, the phase code
    takes one of ``2**n`` values. The phase autoencoder of thesis Table 4.4 uses
    ``n = 2``, so four phase codes are representable.
    """
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1.0):
    """Binarize weights to ``{-H, +H}`` with a straight-through gradient."""
    return H * binary_tanh(W / H)
