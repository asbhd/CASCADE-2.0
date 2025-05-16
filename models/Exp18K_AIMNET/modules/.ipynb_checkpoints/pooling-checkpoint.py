import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from modules.segment import segment_ops_by_name
from kgcnn.ops.scatter import tensor_scatter_nd_ops_by_name

ks = tf.keras

@ks.utils.register_keras_serializable(package='module', name='PoolingEmbedding')
class PoolingEmbedding(GraphBaseLayer):
    """Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    ::obj`tf.Tensor`.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingEmbedding, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Embedding tensor of shape (batch, [N], F)
    
        Returns:
            tf.Tensor: Pooled node features of shape (batch, F)
        """
        self.assert_ragged_input_rank(inputs)
        nod, batchi = inputs[0].values, inputs[0].value_rowids()
        atom_index = inputs[1].values
        n_pro = inputs[2].values
        out = segment_ops_by_name(self.pooling_method, nod, batchi,atom_index,n_pro)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbedding, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingNodes = PoolingEmbedding
PoolingGlobalEdges = PoolingEmbedding



