import tensorflow as tf
import numpy as np
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.layers.conv.painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeDirectionNormalized, CosCutOffEnvelope, \
    NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP

from modules.pooling import PoolingNodes
from modules.repeat import Repeat
from modules.bessel_basis import BesselBasisLayer
from modules.kernel import MaternKernelFn
from modules.mae_callback import val_mae

from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.model.utils import update_model_kwargs
ks = tf.keras

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tf.keras.backend.set_floatx('float64')

num_inducing_points=250
inducing_points = np.load('inducing_index_points_250.npy')

model_default = {
    "name": "PAiNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None,), "name": "atom_index", "dtype": "int32", "ragged": True},
        {"shape": (None,1), "name": "n_pro", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 256, "output_dim": 256}},
    "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
    "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    "pooling_args": {"pooling_method": "mean"},
    "conv_args": {"units": 256, "cutoff": None},
    "update_args": {"units": 256},
    "equiv_normalization": False, "node_normalization": False,
    "depth": 6,
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [256, 1], "activation": ["swish", "linear"]}, "Training":True
}

@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               equiv_initialize_kwargs: dict = None,
               bessel_basis: dict = None,
               depth: int = None,
               pooling_args: dict = None,
               conv_args: dict = None,
               update_args: dict = None,
               equiv_normalization: bool = None,
               node_normalization: bool = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               Training=True
               ):

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    atom_index_input = ks.layers.Input(**inputs[3])
    n_pro_input = ks.layers.Input(**inputs[4])
    
    z = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=True)(node_input)
    
    equiv_input = EquivariantInitialize(**equiv_initialize_kwargs)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])

    n = z
   
    out = PoolingNodes(**pooling_args)([n,atom_index_input,n_pro_input])
    out = Repeat()([out,n_pro_input],training=Training)
    out = tfp.layers.VariationalGaussianProcess(
            num_inducing_points=num_inducing_points,
            kernel_provider=MaternKernelFn(),
            event_shape=[1],
            inducing_index_points_initializer=tf.constant_initializer(inducing_points),
            unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(np.array(0.5).astype('float32'))))(out)
        
    model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input,atom_index_input,n_pro_input], outputs=out)
    
    return model