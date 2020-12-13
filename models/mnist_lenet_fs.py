import os
import numpy as np
import tensorflow as tf
from utils.logger import get_logger, Progbar
from utils.prepro_text import remap_labels, dense_to_one_hot
from utils.data_funcs import compute_ensemble_accuracy


def mnist_lenet(inputs, training=False, reuse=tf.AUTO_REUSE, name="lenet"):  # used for mnist
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        x = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        # first convolutional layer
        conv1 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                 use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(), name="conv1",
                                 reuse=tf.AUTO_REUSE, bias_initializer=tf.constant_initializer(0.05))
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool1")
        drop1 = tf.layers.dropout(pool1, rate=0.5, training=training, name="dropout1")
        # second convolutional layer
        conv2 = tf.layers.conv2d(drop1, filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                 use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(), name="conv2",
                                 reuse=tf.AUTO_REUSE, bias_initializer=tf.constant_initializer(0.05))
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool2")
        drop2 = tf.layers.dropout(pool2, rate=0.5, training=training, name="dropout2")
        # third convolutional layer
        conv3 = tf.layers.conv2d(drop2, filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                 use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(), name="conv3",
                                 reuse=tf.AUTO_REUSE, bias_initializer=tf.constant_initializer(0.05))
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool3")
        drop3 = tf.layers.dropout(pool3, rate=0.5, training=training, name="dropout3")
        # flatten
        features = tf.layers.flatten(drop3, name="flatten")
        return features


def fc_layer(features, labels, label_size, training=False, reuse=tf.AUTO_REUSE, name="fc_layer"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        # first dense layer
        dense1 = tf.layers.dense(features, units=512, activation=tf.nn.relu, use_bias=True, reuse=tf.AUTO_REUSE,
                                 name="dense1")
        dense1_drop = tf.layers.dropout(dense1, rate=0.5, training=training, name="dense_dropout")
        # second dense layer
        logits = tf.layers.dense(dense1_drop, units=label_size, activation=None, use_bias=True, reuse=tf.AUTO_REUSE,
                                 name="logits")
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        pred_labels = tf.argmax(logits, 1)
        return pred_labels, cost


class MnistLeNet:
    def __init__(self, num_class, num_classifier, nary_ecoc, ckpt_path):
        self.ckpt_path, self.num_class = ckpt_path, num_class
        self.num_classifier = num_classifier
        self.nary_ecoc = nary_ecoc
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.ckpt_path + "log.txt")
        self.batch_size, self.epochs, self.lr, self.lr_decay = 200, 10, 0.001, 0.9
        with tf.Graph().as_default():
            self._build_model()
            self.logger.info("total params: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                               tf.trainable_variables()])))
            self._init_session()

    def _init_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

    def save_session(self, steps):
        self.saver.save(self.sess, self.ckpt_path + "mnist_lenet", global_step=steps)

    def restore_last_session(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No pre-trained model at {}".format(self.ckpt_path))

    def close_session(self):
        self.sess.close()

    def _get_feed_dict(self, images, labels, training=False):
        feed_dict = {self.inputs: images}
        for i in range(self.num_classifier):
            feed_dict[self.labels[i]] = labels[i]
        feed_dict[self.training] = training
        return feed_dict

    def _build_model(self):
        # add placeholders
        self.inputs = tf.placeholder(tf.float32, shape=[None, 784], name="input_images")
        self.labels = []
        for i in range(self.num_classifier):
            self.labels.append(tf.placeholder(tf.float32, shape=(None, self.num_class), name="labels_%d" % i))
        self.training = tf.placeholder(tf.bool, shape=[], name="is_training")
        # build model
        x = tf.reshape(self.inputs, shape=[-1, 28, 28, 1])
        features = mnist_lenet(x, self.training)
        self.pred_labels = []
        self.cost = []
        for i in range(self.num_classifier):
            pred_labels, loss = fc_layer(features, self.labels[i], self.num_class, self.training, name="fc_%d" % i)
            pred_labels = tf.expand_dims(pred_labels, axis=-1)
            self.pred_labels.append(pred_labels)
            self.cost.append(loss)
        self.pred_labels = tf.concat(self.pred_labels, axis=-1)
        self.cost = tf.add_n(self.cost)

        # build optimizer and training operation
        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=self.lr_decay)
        self.train_op = optimizer.minimize(self.cost)

    def train(self, train_dataset, test_dataset):
        global_test_acc = 0.0
        global_step = 0
        test_imgs, test_labels = test_dataset.images, test_dataset.labels
        self.logger.info("start training...")
        for epoch in range(1, self.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.epochs))
            num_batches = train_dataset.num_examples // self.batch_size
            prog = Progbar(target=num_batches)
            prog.update(0, [("Global Step", 0), ("Train Loss", 0.0), ("Train Acc", 0.0), ("Test Loss", 0.0),
                            ("Test Acc", 0.0)])
            for i in range(num_batches):
                global_step += 1
                train_imgs, train_labels = train_dataset.next_batch(self.batch_size)
                b_labels = []
                for j in range(self.num_classifier):
                    ecoc_array = self.nary_ecoc[:, j]
                    b_lbs = remap_labels(train_labels.copy(), ecoc_array)
                    b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                    b_labels.append(b_lbs)
                feed_dict = self._get_feed_dict(train_imgs, b_labels, True)
                _, pred_labels, loss = self.sess.run([self.train_op, self.pred_labels, self.cost], feed_dict=feed_dict)
                acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, train_labels)
                if global_step % 100 == 0:
                    y_labels = []
                    for j in range(self.num_classifier):
                        ecoc_array = self.nary_ecoc[:, j]
                        b_lbs = remap_labels(test_labels.copy(), ecoc_array)
                        b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                        y_labels.append(b_lbs)
                    feed_dict = self._get_feed_dict(test_imgs, y_labels)
                    test_pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
                    test_acc = compute_ensemble_accuracy(test_pred_labels, self.nary_ecoc, test_labels)
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc),
                                        ("Test Loss", test_loss), ("Test Acc", test_acc)])
                    if test_acc > global_test_acc:
                        global_test_acc = test_acc
                        self.save_session(global_step)
                else:
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc)])
            y_labels = []
            for j in range(self.num_classifier):
                ecoc_array = self.nary_ecoc[:, j]
                b_lbs = remap_labels(test_labels.copy(), ecoc_array)
                b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                y_labels.append(b_lbs)
            feed_dict = self._get_feed_dict(test_imgs, y_labels)
            test_pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
            test_acc = compute_ensemble_accuracy(test_pred_labels, self.nary_ecoc, test_labels)
            self.logger.info("Epoch: {}, Global Step: {}, Test Loss: {}, Test Accuracy: {}".format(
                epoch, global_step, test_loss, test_acc))

    def test(self, test_dataset, print_info=True):
        self.restore_last_session()
        test_imgs, test_labels = test_dataset.images, test_dataset.labels
        y_labels = []
        for j in range(self.num_classifier):
            ecoc_array = self.nary_ecoc[:, j]
            b_lbs = remap_labels(test_labels.copy(), ecoc_array)
            b_lbs = dense_to_one_hot(b_lbs, self.num_class)
            y_labels.append(b_lbs)
        feed_dict = self._get_feed_dict(test_imgs, y_labels)
        pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
        test_acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, test_dataset.labels)
        if print_info:
            self.logger.info(" -- Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
        return pred_labels
