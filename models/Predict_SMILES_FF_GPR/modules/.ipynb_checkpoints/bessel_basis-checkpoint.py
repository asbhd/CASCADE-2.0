import numpy as np
import tensorflow as tf
from typing import Union
from kgcnn.layers.base import GraphBaseLayer

class BesselBasisLayer(GraphBaseLayer):
    r"""Expand a distance into a Bessel Basis with :math:`l=m=0`, according to
    `Gasteiger et al. (2020) <https://arxiv.org/abs/2011.14115>`_.

    For :math:`l=m=0` the 2D spherical Fourier-Bessel simplifies to
    :math:`\Psi_{\text{RBF}}(d)=a j_0(\frac{z_{0,n}}{c}d)` with roots at :math:`z_{0,n} = n\pi`. With normalization
    on :math:`[0,c]` and :math:`j_0(d) = \sin{(d)}/d` yields
    :math:`\tilde{e}_{\text{RBF}} \in \mathbb{R}^{N_{\text{RBF}}}`:

    .. math::

        \tilde{e}_{\text{RBF}, n} (d) = \sqrt{\frac{2}{c}} \frac{\sin{\left(\frac{n\pi}{c} d\right)}}{d}

    Additionally, applies an envelope function :math:`u(d)` for continuous differentiability on the basis
    :math:`e_{\text{RBF}} = u(d)\tilde{e}_{\text{RBF}}`.
    By Default this is a polynomial of the form:

    .. math::

        u(d) = 1 − \frac{(p + 1)(p + 2)}{2} d^p + p(p + 2)d^{p+1} − \frac{p(p + 1)}{2} d^{p+2},

    where :math:`p \in \mathbb{N}_0` and typically :math:`p=6`.
    """

    def __init__(self, num_radial: int,
                 cutoff: float,
                 envelope_exponent: int = 5,
                 envelope_type: str = "poly",
                 **kwargs):
        r"""Initialize :obj:`BesselBasisLayer` layer.

        Args:
            num_radial (int): Number of radial basis functions to use.
            cutoff (float): Cutoff distance.
            envelope_exponent (int): Degree of the envelope to smoothen at cutoff. Default is 5.
            envelope_type (str): Type of envelope to use. Default is "poly".
        """
        super(BesselBasisLayer, self).__init__(**kwargs)
        # Layer variables
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float64)
        self.envelope_exponent = envelope_exponent
        self.envelope_type = str(envelope_type)

        if self.envelope_type not in ["poly"]:
            raise ValueError("Unknown envelope type '%s' in `BesselBasisLayer`." % self.envelope_type)

        # Initialize frequencies at canonical positions.
        def freq_init(shape, dtype):
            return tf.constant(np.pi * np.arange(1, shape + 1, dtype=np.float64), dtype=dtype)

        self.frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
                                           dtype=tf.float64, initializer=freq_init, trainable=True)
        
    @tf.function
    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))

    def expand_bessel_basis(self, inputs):
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        out = d_cutoff * tf.sin(self.frequencies * d_scaled)
        return out

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: distance

                - distance (tf.RaggedTensor): Edge distance of shape `(batch, [K], 1)`

        Returns:
            tf.RaggedTensor: Expanded distance. Shape is `(batch, [K], num_radial)`.
        """
        return self.map_values(self.expand_bessel_basis, inputs)

    def get_config(self):
        """Update config."""
        config = super(BesselBasisLayer, self).get_config()
        config.update({"num_radial": self.num_radial, "cutoff": self.cutoff,
                       "envelope_exponent": self.envelope_exponent, "envelope_type": self.envelope_type})
        return config