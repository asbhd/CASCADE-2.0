import tensorflow as tf
from keras.layers import Layer
from keras.utils import control_flow_util

class Repeat(Layer):

    def __init__(self, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        
    def call(self, inputs,training=None):
        
        repeats = 1
        single_atom=1
        single_batch=1
        
        if inputs[1].values.shape[0] == single_batch:
            if tf.get_static_value(inputs[1].values[0]) == single_atom:
                repeats = 2

        output = control_flow_util.smart_cond(
            training, lambda: tf.identity(inputs[0]), lambda: tf.repeat(inputs[0], repeats, axis=0))
        
        return output