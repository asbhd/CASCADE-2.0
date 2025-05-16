import tensorflow as tf

@tf.function
def segment_ops_by_name(segment_name: str, data, segment_ids, atom_index, n_pro):
    """Segment operation chosen by string identifier.

    Args:
        segment_name (str): Name of the segment operation.
        data (tf.Tensor): Data tensor that has sorted segments.
        segment_ids (tf.Tensor): IDs of the segments.

    Returns:
        tf.Tensor: reduced segment data with method by segment_name.
    """
    if segment_name in ["segment_mean", "mean", "reduce_mean"]:
        num_segments = tf.reduce_sum(n_pro)
        pool = tf.math.unsorted_segment_mean(data, atom_index, num_segments)
    else:
        raise TypeError("Unknown segment operation, choose: 'segment_mean', 'mean', 'reduce_mean'")
    return pool