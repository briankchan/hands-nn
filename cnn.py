import tensorflow as tf
import misc

from model import save_args
from tfmodel import TFModel


class CNN(TFModel):
    _class_log_path_pattern = "cnn/run{}"

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
        super().__init__(width, height, depth, epochs, batch_size, rate, epsilon, pos_weight)

    def _build_network(self, images):
        """Model function for CNN."""
        with tf.name_scope("network"):
            regularization_scale = 1.

            images = tf.subtract(tf.divide(images, 255 / 2), 1)

            # Input Layer
            input_layer = tf.reshape(images, [-1, self.height, self.width, self.depth])

            # Convolutional Layer and pooling layer #1
            with tf.name_scope("conv_1"):
                conv1 = tf.layers.conv2d(
                    name="conv1",
                    inputs=input_layer,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                pool1 = tf.layers.max_pooling2d(name="pool1", inputs=conv1, pool_size=2, strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope("conv_2"):
                conv2 = tf.layers.conv2d(
                    name="conv2",
                    inputs=pool1,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                pool2 = tf.layers.max_pooling2d(name="pool2", inputs=conv2, pool_size=2, strides=2)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.name_scope("conv_3"):
                conv3 = tf.layers.conv2d(
                    name="conv3",
                    inputs=pool2,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    activation=misc.prelu)
                pool3 = tf.layers.max_pooling2d(name="pool3", inputs=conv3, pool_size=2, strides=1, padding="same")

            # Convolutional Layer #4 and Pooling Layer #4
            with tf.name_scope("conv_4"):
                conv4 = tf.layers.conv2d(
                    name="conv4",
                    inputs=pool3,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                pool4 = tf.layers.max_pooling2d(name="pool4", inputs=conv4, pool_size=2, strides=1, padding="same")

            # Convolutional Layer #5 and Pooling Layer #5
            with tf.name_scope("conv_5"):
                conv5 = tf.layers.conv2d(
                    name="conv5",
                    inputs=pool4,
                    filters=32,
                    kernel_size=17,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                pool5 = tf.layers.max_pooling2d(name="pool5", inputs=conv5, pool_size=2, strides=1, padding="same")

            with tf.name_scope("small_conv_1"):
                small_conv1 = tf.layers.conv2d(
                    name="small_conv1",
                    inputs=pool5,
                    filters=64,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)

                dropout1 = tf.layers.dropout(
                    name="dropout1",
                    inputs=small_conv1,
                    rate=0.1
                )

            with tf.name_scope("small_conv_2"):
                small_conv2 = tf.layers.conv2d(
                    name="small_conv2",
                    inputs=dropout1,
                    filters=64,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)

                dropout2 = tf.layers.dropout(
                    name="dropout2",
                    inputs=small_conv2,
                    rate=0.1
                )

            with tf.name_scope("deconv"):
                deconv = tf.layers.conv2d_transpose(
                    name="deconv",
                    inputs=dropout2,
                    filters=1,
                    kernel_size=16,
                    strides=4,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=None)

            # tf.contrib.layers.summarize_activations()
            # # tf.contrib.layers.summarize_variables()
            # # tf.contrib.layers.summarize_weights()
            # # tf.contrib.layers.summarize_biases()
            # tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
            # tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)

            # print(input_layer.shape)
            # print(1)
            # print(conv1.shape)
            # print(pool1.shape)
            # print(2)
            # print(conv2.shape)
            # print(pool2.shape)
            # print(3)
            # print(conv3.shape)
            # print(pool3.shape)
            # print(4)
            # print(conv4.shape)
            # print(pool4.shape)
            # print(5)
            # print(conv5.shape)
            # print(pool5.shape)
            # print("small", 1)
            # print(small_conv1.shape)
            # print("small", 2)
            # print(small_conv2.shape)
            # print(deconv.shape)

            # first, second = tf.unstack(deconv, axis=3)
            # labels = tf.greater(first, second)  # first is True, second is False

            logits = tf.squeeze(deconv, axis=3)

            return logits
