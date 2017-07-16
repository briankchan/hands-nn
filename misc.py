import tensorflow as tf

def prelu(input, weights_init=tf.constant_initializer(0.25)):
    alphas=tf.get_variable('alpha', input.get_shape()[-1],
                           initializer=weights_init,
                           dtype=tf.float32)
    pos = tf.nn.relu(input)

    neg = alphas * (input - abs(input)) * 0.5

    return pos + neg

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class classproperty(object):
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
