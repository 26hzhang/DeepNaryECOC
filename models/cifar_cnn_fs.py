import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import utils
from utils.logger import get_logger, Progbar
from keras.preprocessing.image import ImageDataGenerator
from utils.prepro_text import dense_to_one_hot, remap_labels
from utils.data_funcs import compute_ensemble_accuracy


def batch_dataset(images, labels, batch_size):
    total = labels.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield images[start:end], labels[start:end]


def cifar_conv_block(inputs, filters, kernel_size, activation, pool_size, weight_decay, drop_rate,
                     training=False, name="conv_block"):
    with tf.variable_scope(name, dtype=tf.float32):
        conv1 = tf.layers.conv2d(inputs, filters=filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                                 padding="same", use_bias=True, activation=activation, name="conv1",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                 trainable=False)
        bn1 = tf.layers.batch_normalization(conv1, name="bn1", trainable=False)
        drop1 = tf.layers.dropout(bn1, rate=drop_rate, training=training, name="dropout")
        conv2 = tf.layers.conv2d(drop1, filters=filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                                 padding="same", use_bias=True, activation=activation, name="conv2",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                 trainable=False)
        bn2 = tf.layers.batch_normalization(conv2, name="bn2", trainable=False)
        pool2 = tf.layers.max_pooling2d(bn2, pool_size=pool_size, strides=2, padding="valid", name="max_pool2")
        return pool2


def cifar_cnn(inputs, training=False, reuse=tf.AUTO_REUSE, name="cifar_cnn"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        conv0 = cifar_conv_block(inputs, filters=32, kernel_size=3, activation=tf.nn.elu, pool_size=2,
                                 weight_decay=5e-4, drop_rate=0.3, training=training, name="conv_block0")
        conv1 = cifar_conv_block(conv0, filters=64, kernel_size=3, activation=tf.nn.elu, pool_size=2,
                                 weight_decay=5e-4, drop_rate=0.4, training=training, name="conv_block1")
        conv2 = cifar_conv_block(conv1, filters=128, kernel_size=3, activation=tf.nn.elu, pool_size=2,
                                 weight_decay=5e-4, drop_rate=0.4, training=training, name="conv_block2")
        conv3 = cifar_conv_block(conv2, filters=256, kernel_size=3, activation=tf.nn.elu, pool_size=2,
                                 weight_decay=5e-4, drop_rate=0.4, training=training, name="conv_block3")
        drop1 = tf.layers.dropout(conv3, rate=0.5, training=training, name="drop1")
        features = tf.layers.flatten(drop1, name="flatten")
        return features


def fc_layer(features, labels, label_size, training=False, reuse=tf.AUTO_REUSE, name="output_layer"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        dense1 = tf.layers.dense(features, units=512, use_bias=True, activation=tf.nn.relu, name="dense1",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
        bn1 = tf.layers.batch_normalization(dense1, name="bn1")
        drop2 = tf.layers.dropout(bn1, rate=0.5, training=training, name="drop2")
        logits = tf.layers.dense(drop2, units=label_size, use_bias=True, activation=None, name="output",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
        pred_labels = tf.argmax(logits, axis=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)) + tf.add_n(
            tf.losses.get_regularization_losses())
        return pred_labels, cost


class CifarCNN:
    def __init__(self, num_class, num_classifier, nary_ecoc, ckpt_path, pretrained_model):
        tf.reset_default_graph()
        self.ckpt_path, self.num_class = ckpt_path, num_class
        self.num_classifier, self.nary_ecoc = num_classifier, nary_ecoc
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.ckpt_path + "log.txt")
        # self.batch_size, self.epochs = 200, epochs
        self.learning_rate, self.lr_decay, self.grad_clip = 0.002, 0.05, 5.0  # adam
        with tf.Graph().as_default():
            self._build_model()
            self.logger.info("total params: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                               tf.trainable_variables()])))
            self._init_session()
            self._restore_part_weights(pretrained_model)

    def _init_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

    def _restore_part_weights(self, model_path):
        variables = slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if "conv_block" in v.name]
        tf.train.Saver(variables_to_restore).restore(self.sess, model_path)

    def save_session(self, steps):
        self.saver.save(self.sess, self.ckpt_path + "cifar_cnn", global_step=steps)

    def restore_last_session(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No pre-trained model at {}".format(self.ckpt_path))

    def close_session(self):
        self.sess.close()

    @staticmethod
    def normalize(x_train, x_test):
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)
        return x_train, x_test

    @staticmethod
    def normalize_10_production(x):
        mean, std = 120.707, 64.15  # statistics from training dataset
        return (x - mean) / (std + 1e-7)

    @staticmethod
    def normalize_100_production(x):
        mean, std = 121.936, 68.389
        return (x - mean) / (std + 1e-7)

    def _get_feed_dict(self, images, labels, lr=None, training=None):
        feed_dict = {self.inputs: images}
        for i in range(self.num_classifier):
            feed_dict[self.labels[i]] = labels[i]
        if lr is not None:
            feed_dict[self.lr] = lr
        feed_dict[self.training] = training
        return feed_dict

    def _build_model(self):
        # add placeholders
        self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="inputs")
        self.labels = []
        for i in range(self.num_classifier):
            self.labels.append(tf.placeholder(tf.float32, shape=(None, self.num_class), name="labels_%d" % i))
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.training = tf.placeholder(tf.bool, shape=[], name="training")

        # build model
        features = cifar_cnn(self.inputs, self.training)
        self.pred_labels, self.cost = [], []
        for i in range(self.num_classifier):
            pred_labels, loss = fc_layer(features, self.labels[i], self.num_class, self.training, name="out_%d" % i)
            pred_labels = tf.expand_dims(pred_labels, axis=-1)
            self.pred_labels.append(pred_labels)
            self.cost.append(loss)
        self.pred_labels = tf.concat(self.pred_labels, axis=-1)
        self.cost = tf.add_n(self.cost)

        # build optimizer and training operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads, vs = zip(*optimizer.compute_gradients(self.cost))
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, vs))

    def train(self, x_train, y_train, x_test, y_test, batch_size=200, epochs=10):
        x_train, x_test = self.normalize(x_train, x_test)

        self.logger.info("data augmentation...")
        datagen = ImageDataGenerator(featurewise_center=True, samplewise_center=False, horizontal_flip=True, cval=0.0,
                                     featurewise_std_normalization=False, preprocessing_function=None, rescale=None,
                                     samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06,
                                     rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0,
                                     zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest',  vertical_flip=False,
                                     data_format="channels_last")
        datagen.fit(x_train)
        x_aug, y_aug = x_train.copy(), y_train.copy()
        x_aug = datagen.flow(x_aug, np.zeros(x_train.shape[0]), batch_size=x_train.shape[0], shuffle=False).next()[0]
        x_train, y_train = np.concatenate((x_train, x_aug)), np.concatenate((y_train, y_aug))
        self.logger.info("start training...")
        global_step, lr, global_test_acc = 0, self.learning_rate, 0.0
        num_batches = x_train.shape[0] // batch_size

        for epoch in range(1, epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, epochs))
            x_train, y_train = utils.shuffle(x_train, y_train, random_state=0)  # shuffle training dataset
            prog = Progbar(target=num_batches)
            prog.update(0, [("Global Step", 0), ("Train Loss", 0.0), ("Train Acc", 0.0), ("Test Loss", 0.0),
                            ("Test Acc", 0.0)])
            for i, (batch_imgs, batch_labels) in enumerate(batch_dataset(x_train, y_train, batch_size)):
                global_step += 1
                b_labels = []
                for j in range(self.num_classifier):
                    ecoc_array = self.nary_ecoc[:, j]
                    b_lbs = remap_labels(batch_labels.copy(), ecoc_array)
                    b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                    b_labels.append(b_lbs)
                feed_dict = self._get_feed_dict(batch_imgs, b_labels, lr, True)
                _, pred_labels, loss = self.sess.run([self.train_op, self.pred_labels, self.cost], feed_dict=feed_dict)
                acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, batch_labels)
                if global_step % 200 == 0:
                    y_labels = []
                    for j in range(self.num_classifier):
                        ecoc_array = self.nary_ecoc[:, j]
                        b_lbs = remap_labels(y_test.copy(), ecoc_array)
                        b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                        y_labels.append(b_lbs)
                    feed_dict = self._get_feed_dict(x_test, y_labels)
                    test_pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
                    test_acc = compute_ensemble_accuracy(test_pred_labels, self.nary_ecoc, y_test)
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc),
                                        ("Test Loss", test_loss), ("Test Acc", test_acc)])
                    if test_acc > global_test_acc:
                        global_test_acc = test_acc
                        self.save_session(global_step)
                else:
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc)])
            if epoch > 10:
                lr = self.learning_rate / (1 + (epoch - 10) * self.lr_decay)
            y_labels = []
            for j in range(self.num_classifier):
                ecoc_array = self.nary_ecoc[:, j]
                b_lbs = remap_labels(y_test.copy(), ecoc_array)
                b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                y_labels.append(b_lbs)
            feed_dict = self._get_feed_dict(x_test, y_labels)
            test_pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
            test_acc = compute_ensemble_accuracy(test_pred_labels, self.nary_ecoc, y_test)
            self.logger.info("Epoch: {}, Global Step: {}, Test Loss: {}, Test Accuracy: {}".format(
                epoch, global_step, test_loss, test_acc))

    def test(self, x_test, y_test,  print_info=True):
        self.restore_last_session()
        if self.num_class > 10:
            x_test = self.normalize_100_production(x_test)
        else:
            x_test = self.normalize_10_production(x_test)
        y_labels = []
        for j in range(self.num_classifier):
            ecoc_array = self.nary_ecoc[:, j]
            b_lbs = remap_labels(y_test.copy(), ecoc_array)
            b_lbs = dense_to_one_hot(b_lbs, self.num_class)
            y_labels.append(b_lbs)
        feed_dict = self._get_feed_dict(x_test, y_labels)
        pred_labels, test_loss = self.sess.run([self.pred_labels, self.cost], feed_dict=feed_dict)
        test_acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, y_test)
        if print_info:
            self.logger.info(" -- Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
        return pred_labels
