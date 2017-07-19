import tensorflow as tf

from models.slim.nets import inception_v2
from tensorflow.contrib import slim

from model import save_args
from tfmodel import TFModel

INCEPTION_CHECKPOINT = "./inception_v2.ckpt"

class CNN(TFModel):
    _class_log_path_pattern = "cnn_inception/run{}"

    @save_args
    def __init__(self,
                 width=640,
                 height=480,
                 depth=3,
                 epochs=1,
                 batch_size=12,
                 rate=0.0001,
                 epsilon=1e-8,
                 pos_weight=10):
        super().__init__()

    def _build_network(self, images):
        with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
            net, end_points = inception_v2.inception_v2_base(images, final_endpoint='Mixed_3c')

        net = slim.avg_pool2d(net, [7, 7], stride=1, scope="MaxPool_0a_7x7")
        net = slim.dropout(net,
                           0.8, scope='Dropout_0b')
        net = slim.conv2d(net, 1, [1, 1], activation_fn=None,
                          normalizer_fn=None)  # , scope='Conv2d_0c_1x1'

        net = tf.pad(net, [[0, 0], [3, 3], [3, 3], [0, 0]])

        net = tf.layers.conv2d_transpose(
            name="deconv",
            inputs=net,
            filters=1,
            kernel_size=16,
            strides=8,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1.),
            activation=None)

        return tf.squeeze(net, axis=3)

    def _initialize_model(self):
        self.session.run(tf.global_variables_initializer())
        vars = [var for var in tf.global_variables() if var.name.startswith("InceptionV2")]
        saver = tf.train.Saver(vars)
        saver.restore(self.session, INCEPTION_CHECKPOINT)
