import os
import keras
import numpy as np
import sklearn.utils
import tensorflow as tf
from utils.logger import get_logger, Progbar


def batch_dataset(images, labels, batch_size):
    total = labels.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield images[start:end], labels[start:end]


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        if groups == 1:
            conv_ = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv_ = tf.concat(axis=3, values=output_groups)
        bias = tf.nn.bias_add(conv_, biases)
        relu = tf.nn.relu(bias, name=scope.name)
        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        return tf.nn.relu(act) if relu else act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


class FlowerAlexNet:
    # https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d
    def __init__(self, num_class, ckpt_path, skip_layer=None, update_weight=False):
        self.num_class, self.ckpt_path = num_class, ckpt_path
        self.update_weight = update_weight
        self.lr = 0.01
        self.skip_layer = skip_layer if skip_layer is not None else ['fc8', 'fc7']
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.ckpt_path + "log.txt")
        with tf.Graph().as_default():
            self._build_model()
            self.logger.info("total params: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                               tf.trainable_variables()])))
            self._init_session()
            self.load_initial_weights()

    def _init_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

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

    def get_conv_variable(self):
        v = self.sess.graph.get_tensor_by_name("conv1/weights:0")
        return self.sess.run(v)

    def _build_model(self):
        # add placeholders
        self.inputs = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name="inputs")
        self.labels = tf.placeholder(tf.float32, shape=(None, self.num_class), name="labels")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # build model
        conv1 = conv(self.inputs, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, keep_prob=self.keep_prob)
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, keep_prob=self.keep_prob)
        self.logits = fc(dropout7, 4096, self.num_class, relu=False, name='fc8')

        with tf.name_scope("cross_ent"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                                  labels=self.labels))

        with tf.name_scope("accuracy"):
            self.pred_labels = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1)),
                                                   dtype=tf.float32))

        with tf.name_scope("train"):
            # List of trainable variables of the layers we want to train
            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.skip_layer]
            # Get gradients of all trainable variables
            gradients = tf.gradients(self.cost, var_list)
            gradients = list(zip(gradients, var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    def load_initial_weights(self):
        # Load the weights into memory
        weights_dict = np.load("datasets/raw/flower102/bvlc_alexnet.npy", encoding='bytes').item()
        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=self.update_weight)
                            self.sess.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=self.update_weight)
                            self.sess.run(var.assign(data))

    def train(self, x_train, y_train, x_test, y_test, batch_size=100, epochs=20, store_conv=False, store_step=50):
        x_train = x_train.astype(dtype=np.float32)
        x_test = x_test.astype(dtype=np.float32)
        y_train = y_train.astype(dtype=np.int32)
        y_test = y_test.astype(dtype=np.int32)
        y_train = keras.utils.to_categorical(y_train, self.num_class)
        y_test = keras.utils.to_categorical(y_test, self.num_class)

        if store_conv and not os.path.exists(self.ckpt_path + "conv_weights/"):
            os.makedirs(self.ckpt_path + "conv_weights/")

        self.logger.info("start training...")
        global_step, global_test_acc = 0, 0.0
        num_batches = x_train.shape[0] // batch_size
        for epoch in range(1, epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, epochs))
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)  # shuffle training dataset
            prog = Progbar(target=num_batches)
            prog.update(0, [("Global Step", 0), ("Train Loss", 0.0), ("Train Acc", 0.0), ("Test Loss", 0.0),
                            ("Test Acc", 0.0)])
            for i, (batch_imgs, batch_labels) in enumerate(batch_dataset(x_train, y_train, batch_size)):
                global_step += 1
                feed_dict = {self.inputs: batch_imgs, self.labels: batch_labels, self.keep_prob: 0.5}
                _, loss, acc = self.sess.run([self.train_op, self.cost, self.accuracy], feed_dict=feed_dict)
                if global_step % 20 == 0:
                    feed_dict = {self.inputs: x_test, self.labels: y_test, self.keep_prob: 1.0}
                    test_loss, test_acc = self.sess.run([self.cost, self.accuracy], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc),
                                        ("Test Loss", test_loss), ("Test Acc", test_acc)])
                    if test_acc > global_test_acc:
                        global_test_acc = test_acc
                        self.save_session(global_step)
                else:
                    prog.update(i + 1, [("Global Step", int(global_step)), ("Train Loss", loss), ("Train Acc", acc)])
                if store_conv and global_step % store_step == 0:
                    conv_weight = self.get_conv_variable()
                    np.savez_compressed(self.ckpt_path + "conv_weights/conv_{}.npz".format(global_step),
                                        embeddings=conv_weight)

            feed_dict = {self.inputs: x_test, self.labels: y_test, self.keep_prob: 1.0}
            test_loss, test_acc = self.sess.run([self.cost, self.accuracy], feed_dict=feed_dict)
            self.logger.info("Epoch: {}, Global Step: {}, Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(
                epoch, global_step, test_loss, test_acc * 100.0))

    def test(self, x_test, y_test, print_info=True):
        self.restore_last_session()
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:
            y_test = keras.utils.to_categorical(y_test, self.num_class)
        feed_dict = {self.inputs: x_test, self.labels: y_test, self.keep_prob: 1.0}
        pred_labels, test_loss, test_acc = self.sess.run([self.pred_labels, self.cost, self.accuracy],
                                                         feed_dict=feed_dict)
        if print_info:
            self.logger.info(" -- Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_acc * 100))
        return np.reshape(pred_labels, newshape=(pred_labels.shape[0], 1))
