import tensorflow as tf

import misc
from convolutional_lstm import ConvLSTMCell
from model import save_args
from tfmodel import TFModel


class LSTM(TFModel):
    _class_log_path_pattern = "lstm/run{}"

    @save_args
    def __init__(self,
                 width=640,
                 height=480,
                 depth=3,
                 regular_conv_layers=2,
                 large_conv_size=17,
                 large_conv_kernels=32,
                 lstm_kernels=20,
                 lstm_in_drop_rate=0.1,
                 lstm_out_drop_rate=0.1,
                 small_conv_drop_rate=0.1,
                 epochs=1,
                 batch_size=12,
                 rate=0.0001,
                 epsilon=1e-8,
                 pos_weight=10):
        super().__init__(width, height, depth, epochs, batch_size, rate, epsilon, pos_weight)

    @property
    def _train_feed_dict(self):
        return {
            self.lstm_in_dropout: self.lstm_in_drop,
            self.lstm_out_dropout: self.lstm_out_drop,
            self.small_conv_dropout: self.small_conv_drop
        }

    def _build_model(self):
        self.config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.step = tf.Variable(0, trainable=False, name="step")

            self.images, \
            self.target_labels, \
            self.prev_output, \
            self.lstm_initial_state, \
            self.lstm_in_dropout, \
            self.lstm_out_dropout, \
            self.small_conv_dropout = self._build_inputs()

            self.logits = self._build_network(
                self.images,
                self.prev_output,
                self.lstm_initial_state,
                self.lstm_in_dropout,
                self.lstm_out_dropout,
                self.small_conv_dropout)
            self.pred_labels = tf.greater(self.logits, 0)

            self.loss = self._build_loss(self.logits, self.target_labels, self.pos_weight)
            self.optimizer = self._build_optimizer(self.loss, self.rate, self.epsilon, self.step)

            self.confusion_matrix = self._build_evaluator(self.target_labels, self.pred_labels)

            self.reset()

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def _build_inputs(self):
        inputs = super()._build_inputs()
        prev_output = tf.Variable(tf.zeros([self.height, self.width]), dtype=tf.bool, trainable=False, name="prev_output")
        lstm_initial_state = tf.Variable(tf.zeros([self.height, self.width, self.depth]), trainable=False, name="lstm_initial_state")
        lstm_in_dropout = tf.Variable(0, trainable=False, name="lstm_in_dropout")
        lstm_out_dropout = tf.Variable(0, trainable=False, name="lstm_out_dropout")
        small_conv_dropout = tf.Variable(0, trainable=False, name="small_conv_dropout")
        return inputs + (prev_output, lstm_initial_state, lstm_in_dropout, lstm_out_dropout, small_conv_dropout)

    def _build_network(
            self, 
            images, 
            prev_output, 
            lstm_initial_state, 
            lstm_in_dropout,
            lstm_out_dropout, 
            small_conv_dropout):
        """Model function for CNN."""
        with tf.name_scope("network"):
            regularization_scale = 1.

            # rescale to -1 to 1
            images = images / (255 / 2) - 1

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

            pool = pool2
            for i in range(3, 3 + self.regular_conv_layers):
                with tf.name_scope("conv_{}".format(i)):
                    conv = tf.layers.conv2d(
                        name="conv{}".format(i),
                        inputs=pool,
                        filters=32,
                        kernel_size=5,
                        padding="same",
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                        activation=misc.prelu)
                    pool = tf.layers.max_pooling2d(name="pool{}".format(i), inputs=conv, pool_size=2, strides=1,
                                                   padding="same")

            # Convolutional Layer #5 and Pooling Layer #5
            with tf.name_scope("large_conv"):
                large_conv = tf.layers.conv2d(
                    name="large_conv",
                    inputs=pool,
                    filters=self.large_conv_kernels,
                    kernel_size=self.large_conv_size,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                large_pool = tf.layers.max_pooling2d(name="large_pool", inputs=large_conv, pool_size=2, strides=1, padding="same")

            with tf.name_scope("prev_output_conv_1"):
                prev_output_conv1 = tf.layers.conv2d(
                    name="prev_output_conv1",
                    inputs=prev_output,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                prev_output_pool1 = tf.layers.max_pooling2d(name="prev_output_pool1", inputs=prev_output_conv1, pool_size=2, strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope("prev_output_conv_2"):
                prev_output_conv2 = tf.layers.conv2d(
                    name="prev_output_conv2",
                    inputs=prev_output_pool1,
                    filters=32,
                    kernel_size=5,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)
                prev_output_pool2 = tf.layers.max_pooling2d(name="prev_output_pool2", inputs=prev_output_conv2, pool_size=2, strides=2)

            pool = prev_output_pool2
            for i in range(3, 3 + self.regular_conv_layers):
                with tf.name_scope("prev_output_conv_{}".format(i)):
                    conv = tf.layers.conv2d(
                        name="prev_output_conv{}".format(i),
                        inputs=pool,
                        filters=32,
                        kernel_size=5,
                        padding="same",
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                        activation=misc.prelu)
                    pool = tf.layers.max_pooling2d(name="prev_output_pool{}".format(i), inputs=conv, pool_size=2, strides=1,
                                                   padding="same")

            lstm_input = tf.concat([large_pool, pool], axis=-1, name="lstm_input")

            with tf.name_scope("lstm"):
                cell = tf.contrib.rnn.DropoutWrapper(
                    ConvLSTMCell(
                        input_shape=(self.height, self.width, 32 + 20),
                        output_channels=self.lstm_kernels,
                        kernel_shape=(1, 1)),
                    input_keep_prob=1-lstm_in_dropout,
                    output_keep_prob=1-lstm_out_dropout)

                lstm, state = tf.contrib.rnn.dynamic_rnn(
                    cell,
                    lstm_input,
                    initial_state=lstm_initial_state,
                    scope="lstm")

            with tf.name_scope("small_conv"):
                small_conv = tf.layers.conv2d(
                    name="small_conv",
                    inputs=lstm,
                    filters=64,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=misc.prelu)

                dropout = tf.layers.dropout(
                    name="dropout1",
                    inputs=small_conv,
                    rate=small_conv_dropout
                )

            with tf.name_scope("deconv"):
                deconv = tf.layers.conv2d_transpose(
                    name="deconv",
                    inputs=dropout,
                    filters=1,
                    kernel_size=16,
                    strides=4,
                    padding="same",
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
                    activation=None)

            logits = tf.squeeze(deconv, axis=3)

            return logits
