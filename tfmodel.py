import random
import time
from math import ceil
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

from model import Model, save_args
from misc import chunks

class TFModel(Model, metaclass=ABCMeta):
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
        # batch size, epochs, train stuff, height/width, depth, pos weight...

    def _build_model(self):
        self.config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.step = tf.Variable(0, trainable=False, name="step")
                self.images, self.target_labels = self._build_inputs()

                self.logits = self._build_network(self.images)
                self.pred_labels = tf.greater(self.logits, 0)

                self.loss = self._build_loss(self.logits, self.target_labels, self.pos_weight)
                self.optimizer = self._build_optimizer(self.loss, self.rate, self.epsilon, self.step)

                self.confusion_matrix = self._build_evaluator(self.target_labels, self.pred_labels)

                self.reset()

                self.summary = tf.summary.merge_all()
                self.saver = tf.train.Saver()

    def _build_inputs(self):
        with tf.name_scope("input"):
            images = tf.placeholder(tf.float32, [None, self.height, self.width, self.depth], "images")
            target_labels = tf.placeholder(tf.bool, [None, self.height, self.width], "target_labels")
            return images, target_labels

    @abstractmethod
    def _build_network(self, images):
        pass

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
        with tf.name_scope("loss"):
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.to_float(labels), pos_weight=pos_weight)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
            return cross_entropy_mean

    def _build_optimizer(self, loss, rate, epsilon, step):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
            return optimizer.minimize(loss, global_step=step, name="optimize")

    def _build_evaluator(self, target, predicted):
        target_flat = tf.reshape(target, [-1])
        predicted_flat = tf.reshape(predicted, [-1])
        return tf.confusion_matrix(target_flat, predicted_flat)

    def _reset_model(self):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.session = tf.Session(graph=self.graph, config=self.config)
                self._initialize_model()

    def _initialize_model(self):
        self.session.run(tf.global_variables_initializer())

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
        num_batches = ceil(len(indices) / self.batch_size)

        for epoch in range(1, epochs+1):
            random.shuffle(indices)
            batches = chunks(indices, batch_size)
            start = time.time()

            for i, frames in enumerate(batches, 1):
                # _, losses = self.session.run([train_op, tf.get_collection('losses')],
                summary, _ = self.session.run([self.summary, self.optimizer],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, tf.train.global_step(self.session, self.step))
                if i % 10 == 0 or i == num_batches:
                    print("Epoch {}: batch {} of {} | {:.3f}s".format(
                        epoch,
                        i,
                        num_batches,
                        time.time() - start), end="\r")
            print()

        writer.close()

    def predict(self, images, indices=None):
        if indices is None:
            indices = range(len(images))
        else:
            indices = np.r_[tuple(indices)]
        images = images[indices]
        # pred like images, but only 1 channel ([count, h, w] vs [count, h, w, rgb=3])
        pred = np.empty(images.shape[:-1], dtype=np.bool)
        start = time.time()
        # batches = chunks(indices, self.batch_size)
        for i, image in enumerate(images):
            pred[i] = self.session.run(self.pred_labels,
                                       {self.images: [image]})[0]
        print("done in {}s".format(time.time() - start))
        return pred

    def test(self, images, expected, indices):
        if indices is None:
            indices = range(len(input))
        else:
            indices = np.r_[tuple(indices)]

        batch_size = 3  # for some reason this runs faster than larger batch sizes

        # images shape: [count, h, w, rgb=3]; use h and w from image
        pred = np.empty((len(indices), images.shape[1], images.shape[2]), dtype=np.bool)
        batches = chunks(indices, batch_size)
        conf_mat = np.zeros([2, 2], dtype=np.int)

        for i, batch in enumerate(batches):
            s = i * batch_size
            pred[s:s+batch_size], cf = self.session.run(
                [self.pred_labels, self.confusion_matrix],
                {self.images: images[batch], self.target_labels: expected[batch]})
            conf_mat += cf

        accuracy = conf_mat.diagonal().sum() / conf_mat.sum()
        precision = conf_mat[1,1] / conf_mat[:,1].sum()
        recall = conf_mat[1,1] / conf_mat[1].sum()
        f1 = 0 if precision == 0 and recall == 0\
             else 2 * precision * recall / (precision + recall)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)
        print("Confusion matrix")
        print(conf_mat)
        return pred

    def _save_model(self, path):
        self.saver.save(self.session, path+"/model.ckpt")

    def _load_model(self, path):
        # with self.graph.as_default():
        #     with self.graph.device("/gpu:0"):
        self.saver.restore(self.session, path+"/model.ckpt")
