import os
import math
import random
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import misc

# from models.slim.nets import inception_v2
from tensorflow.contrib import slim

DATA1 = "data/hands1-3650"
DATA2 = "data/hands2-3650"
IMAGES = DATA1 + "-images-500.npy"
LABELS = DATA1 + "-labels-500.npy"

IMAGES_FULL = DATA1 + "-images.npy"
LABELS_FULL = DATA1 + "-labels.npy"

IMAGES2_FULL = DATA2 + "-images.npy"
LABELS2_FULL = DATA2 + "-labels.npy"

INCEPTION_CHECKPOINT = "./inception_v2.ckpt"

IMG = np.load(IMAGES)
LAB = np.load(LABELS)
TRAIN = range(400)
TEST = range(400, len(IMG))
# TRAIN_IMG = IMG[:400]
# TRAIN_LAB = LAB[:400]
# TEST_IMG = IMG[400:]
# TEST_LAB = LAB[400:]


class Classifier(object):
    def __init__(self, width=640, height=480, depth=3, inception=False):
        self.config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        self.graph = tf.Graph()
        self.reset()

        self.height = height
        self.width = width
        self.depth = depth
        self.inception = inception

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.images = tf.placeholder(tf.float32, [None, height, width, depth], "images")
                self.target_labels = tf.placeholder(tf.bool, [None, height, width], "target_labels")

                if inception:
                    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
                        net, end_points = inception_v2.inception_v2_base(self.images, final_endpoint='Mixed_3c')

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

                    self.logits = tf.squeeze(net, axis=3)

                    self.pred_labels = tf.greater(self.logits, 0)
                else:
                    self.pred_labels, self.logits = self.build_model(self.images)
                    self.saver = tf.train.Saver()

                self.loss = self.calculate_loss(self.logits, self.target_labels)

                self.summary = tf.summary.merge_all()

                self.saver = tf.train.Saver()


    def reset(self):
        self.session = tf.Session(graph=self.graph, config=self.config)
        i = 0
        while True:
            i += 1
            summary_path = "train/run{0}".format(i)
            if not os.path.exists(summary_path) and not glob.glob(summary_path + "-*"):
                break
        self.summary_path = summary_path
        self.run_num = i
        self.it = 0

    def build_model(self, images):
        """Model function for CNN."""

        regularization_scale = 1.

        images = tf.subtract(tf.divide(images, 255 / 2), 1)

        # Input Layer
        input_layer = tf.reshape(images, [-1, self.height, self.width, self.depth])

        # Convolutional Layer and pooling layer #1
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
        conv3 = tf.layers.conv2d(
            name="conv3",
            inputs=pool2,
            filters=32,
            kernel_size=5,
            padding="same",
            activation=misc.prelu)
        pool3 = tf.layers.max_pooling2d(name="pool3", inputs=conv3, pool_size=2, strides=1, padding="same")

        # Convolutional Layer #4 and Pooling Layer #4
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
        conv5 = tf.layers.conv2d(
            name="conv5",
            inputs=pool4,
            filters=32,
            kernel_size=17,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)
        pool5 = tf.layers.max_pooling2d(name="pool5", inputs=conv5, pool_size=2, strides=1, padding="same")

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
        labels = tf.greater(logits, 0)

        return labels, logits

        # # Dense Layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
        #
        # # Logits Layer
        # logits = tf.layers.dense(inputs=dropout, units=10)

        # Calculate Loss (for both TRAIN and EVAL modes)
        # if mode != learn.ModeKeys.INFER:
        #     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        #     loss = tf.losses.softmax_cross_entropy(
        #         onehot_labels=onehot_labels, logits=logits)
        #
        # # Configure the Training Op (for TRAIN mode)
        # if mode == learn.ModeKeys.TRAIN:
        #     train_op = tf.contrib.layers.optimize_loss(
        #         loss=loss,
        #         global_step=tf.contrib.framework.get_global_step(),
        #         learning_rate=0.001,
        #         optimizer="SGD")
        #
        # # Generate Predictions
        # predictions = {
        #     "classes": tf.argmax(
        #         input=logits, axis=1),
        #     "probabilities": tf.nn.softmax(
        #         logits, name="softmax_tensor")
        # }
        #
        # # Return a ModelFnOps object
        # return model_fn_lib.ModelFnOps(
        #     mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def calculate_loss(self, logits, labels, pos_weight=1):
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
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                with tf.name_scope('loss'):
                    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))

                    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.to_float(labels), pos_weight=pos_weight)
                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
                    tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
                    return cross_entropy_mean

    def make_train_op(self, loss, rate, epsilon, initialize=True):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                if self.inception and initialize:
                    vars = [var for var in tf.global_variables() if var.name.startswith("InceptionV2")]
                    saver = tf.train.Saver(vars)

                optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
                op = optimizer.minimize(loss)

                if initialize:
                    self.session.run(tf.global_variables_initializer())
                    if self.inception:
                        saver.restore(self.session, INCEPTION_CHECKPOINT)

                return op

    def train(self, images, labels, indices=None, epochs=1, batch_size=8, rate=0.0001, epsilon=1e-8, pos_weight=10, reset=True):
        if indices is None:
            indices = list(range(len(images)))
        else:
            indices = list(indices)

        if reset:
            self.reset()
            loss = self.calculate_loss(self.logits, self.target_labels, pos_weight)
            # self.loss = loss
            train_op = self.make_train_op(loss, rate, epsilon, initialize=reset)
            self.train_op = train_op
        else:
            train_op = self.train_op

        writer = tf.summary.FileWriter(self.summary_path, self.graph)

        print("Training")

        for epoch in range(epochs):
            print("===============")
            print("EPOCH", epoch+1)
            print("===============")

            random.shuffle(indices)
            batches = misc.chunks(indices, batch_size)
            i = 0

            for frames in batches:
                self.it += 1
                i += 1
                if i % 10 == 0:
                    print("batch", i)
                # frames = indices[i * batch_size : (i + 1) * batch_size]
                # _, losses = self.session.run([train_op, tf.get_collection('losses')],
                summary, _ = self.session.run([self.summary, train_op],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, self.it)
        writer.close()

    def test(self, images, labels, indices=None):
        if indices is None:
            indices = range(len(images))
            lab = labels
        else:
            indices = np.r_[tuple(indices)]
            lab = labels[indices]
        pred = np.empty_like(lab)
        print("Testing")
        for index, i in zip(indices, range(len(indices))):
            pred[i] = self.session.run(self.pred_labels,
                                       {self.images: [images[index]],
                                        self.target_labels: [labels[index]]})[0]
        errors = (pred != lab).sum()
        print("Avg errors:", errors / len(lab))
        print("Pct errors:", errors / lab.size)

        intersection = (pred * lab).sum()
        pred_sum = pred.sum()
        exp_sum = lab.sum()
        precision = intersection / pred_sum
        recall = intersection.sum() / exp_sum
        f1 = 2 * precision * recall / (precision + recall)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)

        total = lab.size
        print("Confusion matrix")
        print("Predicted hands:", intersection, (pred_sum-intersection))
        print("Predicted not hands", (exp_sum-intersection), (total - pred_sum - exp_sum + intersection))
        print("Total:", total)
        return pred

    def save(self, pathname=None):
        if pathname is None:
            pathname = self.summary_path
        self.saver.save(self.session, pathname+"/model.ckpt")

    def restore(self, run_num=None, pathname=None):
        if pathname is None:
            if run_num is None:
                run_num = self.run_num - 1
            pathname = "train/run{0}".format(run_num)
            if not os.path.exists(pathname):
                pathname = glob.glob(pathname + "-*")[0]

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.saver.restore(self.session, pathname+"/model.ckpt")

m = Classifier(inception=False)

def train(images=IMG, labels=LAB, indices=TRAIN, *args, **kwargs):
    return m.train(images, labels, indices, *args, **kwargs)

def test(images=IMG, labels=LAB, indices=TEST):
    return m.test(images, labels, indices)

def run(img=IMG, lab=LAB, train_indices=TRAIN, test_indices=TEST, *args, **kwargs):
    train(img, lab, train_indices, *args, **kwargs)
    return test(img, lab, test_indices)

def split(ranges, test_chunks, num_chunks):
    test_indices = []
    train_indices = []
    for indices in ranges:
        chunk_size = math.ceil(len(indices) / num_chunks)
        chunks = np.array(list(misc.chunks(indices, chunk_size)))
        test_indices += chunks[test_chunks].tolist()
        train_indices += np.delete(chunks, test_chunks, axis=0).tolist()

    test_indices = np.concatenate(test_indices)
    train_indices = np.concatenate(train_indices)

    return train_indices, test_indices

def split_and_run(images, labels, test_chunks, num_chunks=9, ranges=None, *args, **kwargs):
    if ranges is None:
        ranges = [range(len(images))]

    train_indices, test_indices = split(ranges, test_chunks, num_chunks)

    return run(images, labels, train_indices, test_indices, *args, **kwargs), test_indices

def randsplit_and_run(images, labels, num_chunks=9, num_test_chunks=1, ranges=None, *args, **kwargs):
    test_chunks = np.random.choice(range(num_chunks), num_test_chunks, replace=False)
    return split_and_run(images, labels, test_chunks, num_chunks, ranges, *args, **kwargs)

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def overlay(img, lab, truth):
    img = np.copy(img)
    img[truth, 2] = 255
    img[truth, :2] //= 2

    img[lab, 0] = 255

    intersection = truth*lab
    img[lab - intersection, 1:] //= 2
    return img

def imshow(img):
    plt.imshow(img)
    plt.show()

def labshow(labels, i, images=IMG, truth=LAB, test_indices=TEST):
    if test_indices is None:
        test_indices = range(len(images))
    imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))

def get_all_data():
    # throw out garbage labels at the end
    images1 = np.load(IMAGES_FULL)[:-130]
    labels1 = np.load(LABELS_FULL)[:-130]
    images2 = np.load(IMAGES2_FULL)
    labels2 = np.load(LABELS2_FULL)

    images = np.concatenate([images1, images2])
    labels = np.concatenate([labels1, labels2])

    count1 = len(images1)
    split1 = count1 // 10 * 9
    count2 = len(images2)
    split2 = count2 // 10 * 9

    train_ranges = [range(split1), range(count1, count1+split2)]
    test_ranges = [range(split1, count1), range(count1+split2, count1+count2)]

    return images, labels, train_ranges, test_ranges

def split_run_save(images, labels, train, test, *args, **kwargs):
    train, val = split(train, [5], 9)
    pred = run(images, labels, train, test, *args, **kwargs)
    m.save()
    return pred
