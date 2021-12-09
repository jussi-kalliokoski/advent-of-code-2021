import tensorflow as tf

def reduce_fn(fn, values, initial, **kwargs):
    return tf.while_loop(
        lambda i, current: tf.less(i, tf.shape(values)[0]),
        lambda i, current: (i + 1, fn(current, values[i])),
        [0, initial],
        **kwargs,
    )[1]
