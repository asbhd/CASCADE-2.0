import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
import gzip
import pickle
import pandas as pd
import sys
sys.path.append('modules')
from nfp.preprocessing import MolPreprocessor, GraphSequence
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras import metrics
from keras.metrics import RootMeanSquaredError
import random
import os

# DATA PREPROCESSING

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()

def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

def ragged_const(inp_arr):
    return tf.ragged.constant(np.expand_dims(inp_arr,axis=0), ragged_rank=1)

class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        
        offset = _compute_stacked_offsets(
            batch_data['n_pro'], batch_data['n_atom'])

        offset = np.where(batch_data['atom_index']>=0, offset, 0)
        batch_data['atom_index'] += offset
        
        features = ['node_attributes', 'node_coordinates', 'edge_indices', 'atom_index', 'n_pro']
        for feature in features:
            batch_data[feature] = ragged_const(batch_data[feature])

        del batch_data['n_atom']
        del batch_data['n_bond']
        del batch_data['distance']
        del batch_data['bond']
        del batch_data['node_graph_indices']

        return batch_data

with open('data/processed_inputs.p', 'rb') as f:
    input_data = pickle.load(f)
    
train = pd.read_pickle('data/train.pkl.gz')
valid = pd.read_pickle('data/valid.pkl.gz')
test = pd.read_pickle('data/test.pkl.gz')

y_train = train.Shifts.values
y_valid = valid.Shifts.values
y_test = test.Shifts.values

for i in range(17315):
    y_train[i] -= 99.798111
    y_train[i] /= 50.484337
    
for i in range(2200):
    y_valid[i] -= 99.798111
    y_valid[i] /= 50.484337

batch_size = 64
train_sequence = RBFSequence(input_data['inputs_train'], y_train, batch_size)
valid_sequence = RBFSequence(input_data['inputs_valid'], y_valid, batch_size)
test_sequence = RBFSequence(input_data['inputs_test'], y_test, batch_size)

# MODEL

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.layers.conv.painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistanceEuclidean, BesselBasisLayer, EdgeDirectionNormalized, CosCutOffEnvelope, \
    NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from modules.pooling import PoolingNodes
from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.model.utils import update_model_kwargs
ks = tf.keras

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
    "output_mlp": {"use_bias": [True, True], "units": [256, 1], "activation": ["swish", "linear"]}
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
    out = MLP(**output_mlp)(out)
        
    model = ks.models.Model(inputs=[node_input, xyz_input, bond_index_input,atom_index_input,n_pro_input], outputs=out)
    
    return model

model = make_model()

# TRAINING

def decay_fn(epoch, learning_rate):
    """ Jorgensen decays to 0.96*lr every 100,000 batches, which is approx
    every 70 epochs """
    if (epoch % 35) == 0:
        return 0.96 * learning_rate
    else:
        return learning_rate

lr_decay = LearningRateScheduler(decay_fn)
filepath = "best_model.h5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True, save_freq='epoch', verbose=1)
csv_logger = CSVLogger('log_painn.csv')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=6, min_lr= 1E-4, verbose=1)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0, patience=15,verbose=1)

lr = 7.5E-4
epochs = 250

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mae',metrics=['mse',RootMeanSquaredError()])

model.summary()

start = time.process_time()

model.fit(train_sequence,batch_size=32,epochs=epochs,
          validation_data=valid_sequence,verbose=1,callbacks=[checkpoint, csv_logger, lr_decay, reduce_lr, early_stop])

stop = time.process_time()
print("Time for training: ", str(timedelta(seconds=stop - start)))