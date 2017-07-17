import os
import random
import glob

import numpy as np
import tensorflow as tf
import misc

# from models.slim.nets import inception_v2
from tensorflow.contrib import slim

from model import Model, save_args, chunks


INCEPTION_CHECKPOINT = "./inception_v2.ckpt"

class CNN(Model):
    _class_log_path_pattern = "cnn/run{}"

    @save_args
    def __init__(self,
                 width=640,
                 height=480,
                 depth=3,
                 inception=False,
                 epochs=1,
                 batch_size=16,
                 rate=0.0001,
                 epsilon=1e-8,
                 pos_weight=10):
        super().__init__()

    def _reset_model(self):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.session = tf.Session(graph=self.graph, config=self.config)
                self._initialize_model()
        self.it = 0

    def _build_model(self):
        self.config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.images, self.target_labels = self._build_inputs()

                self.logits = self._build_inception_model(self.images) if self.inception\
                              else self._build_network(self.images)

                self.pred_labels = tf.greater(self.logits, 0)

                self.loss = self._build_loss(self.logits, self.target_labels, self.pos_weight)


                self.optimizer = self._build_optimizer(self.loss, self.rate, self.epsilon)

                self.reset()

                self.summary = tf.summary.merge_all()

                self.saver = tf.train.Saver()

    def _build_inputs(self):
        images = tf.placeholder(tf.float32, [None, self.height, self.width, self.depth], "images")
        target_labels = tf.placeholder(tf.bool, [None, self.height, self.width], "target_labels")
        return images, target_labels

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

            tf.contrib.layers.summarize_activations()
            # tf.contrib.layers.summarize_variables()
            # tf.contrib.layers.summarize_weights()
            # tf.contrib.layers.summarize_biases()
            tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
            tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)

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

    def _build_inception_model(self, images):
        from models.slim.nets import inception_v2
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

    def _build_loss(self, logits, labels, pos_weight=1):
        """Calculate the loss from the logits and the labels.
        Args:
          logits: tensor, float - [batch_size, width, height, num_classes].
              Use vgg_fcn.up as logits.
          labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
              The ground truth of your data.
          weights: numpy array - [num_classes]
              Weighting the loss of each class
              Optional: Prioritize some classes
        Returns:
          loss: Loss tensor of type float.
        """
        with tf.name_scope('loss'):
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.to_float(labels), pos_weight=pos_weight)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
            return cross_entropy_mean

    def _build_optimizer(self, loss, rate, epsilon):
        optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
        return optimizer.minimize(loss)

    def _initialize_model(self):
        self.session.run(tf.global_variables_initializer())
        if self.inception:
            vars = [var for var in tf.global_variables() if var.name.startswith("InceptionV2")]
            saver = tf.train.Saver(vars)
            saver.restore(self.session, INCEPTION_CHECKPOINT)

    def train(self, images, labels, indices=None, epochs=None, batch_size=None):
        if indices is None:
            indices = list(range(len(images)))
        else:
            indices = list(indices)
        if epochs == None:
            epochs = self.epochs
        if batch_size == None:
            batch_size = self.batch_size

        writer = tf.summary.FileWriter(self.log_path, self.graph)

        print("Training")

        for epoch in range(epochs):
            print("===============")
            print("EPOCH", epoch+1)
            print("===============")

            random.shuffle(indices)
            batches = chunks(indices, batch_size)

            for i, frames in enumerate(batches, 1):
                self.it += 1
                if i % 10 == 0:
                    print("batch", i)
                # _, losses = self.session.run([train_op, tf.get_collection('losses')],
                summary, _ = self.session.run([self.summary, self.optimizer],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, self.it)
        writer.close()

    def predict(self, images, indices=None):
        if indices is None:
            indices = range(len(images))
        else:
            indices = np.r_[tuple(indices)]
        images = images[indices]
        # pred like images, but only 1 channel ([count, h, w] vs [count, h, w, rgb=3])
        pred = np.empty(images.shape[:-1], dtype=np.bool)
        print("Testing")
        for i, image in enumerate(images):
            pred[i] = self.session.run(self.pred_labels,
                                       {self.images: [image]})[0]
        return pred

    def _save_model(self, path):
        self.saver.save(self.session, path+"/model.ckpt")

    def _load_model(self, path):
        # with self.graph.as_default():
        #     with self.graph.device("/gpu:0"):
        self.saver.restore(self.session, path+"/model.ckpt")
