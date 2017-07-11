import tensorflow as tf

def prelu(input, weights_init=tf.constant_initializer(0.25)):
    alphas=tf.get_variable('alpha', input.get_shape()[-1],
                           initializer=weights_init,
                           dtype=tf.float32)
    pos = tf.nn.relu(input)

    neg = alphas * (input - abs(input)) * 0.5

    return pos + neg