import tensorflow as tf
import keras
import tensorflow_probability as tfp

# DEFINING KERNEL
class MaternKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaternKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
                initializer=tf.constant_initializer(0),
                dtype=dtype,
                name='amplitude')

        self._length_scale = self.add_variable(
                initializer=tf.constant_initializer(0),
                dtype=dtype,
                name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.MaternThreeHalves(
          amplitude=tf.nn.softplus(1 * self._amplitude),
          length_scale=tf.nn.softplus(1 * self._length_scale)
        )