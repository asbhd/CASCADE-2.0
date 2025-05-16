import tensorflow as tf
import numpy as np

def ragged_input(input_data, split):
    
    split_key = "inputs_" + split

    mol_list = input_data[split_key]

    feature_key = ['node_attributes', 'node_coordinates', 'edge_indices', 'atom_index', 'n_pro']
    dtype_key = {'node_attributes':'float32', 'node_coordinates':'float32', 'edge_indices':'int64',
               'atom_index':'int32','n_pro':'int32'}

    ragged_inputs = []

    for key in feature_key:
        if key == 'n_pro':
            feature_v = [[mol[key]] for mol in mol_list]
        else:
            feature_v = [mol[key] for mol in mol_list]
            
        ragged_feature_tensor = tf.RaggedTensor.from_row_lengths(np.concatenate(feature_v, axis=0),
                                                                np.array([len(x) for x in feature_v], dtype=dtype_key[key]))
        
        ragged_inputs.append(ragged_feature_tensor)
        
    return ragged_inputs