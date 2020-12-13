import os
import math
import numpy as np
import sklearn.utils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.logger import get_logger, Progbar
from utils.prepro_text import batch_iter, dense_to_one_hot, remap_labels
from utils.data_funcs import compute_ensemble_accuracy


def highway_layer(inputs, num_unit, activation, use_bias=True, reuse=tf.AUTO_REUSE, name="highway"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        trans_gate = tf.layers.dense(inputs, units=num_unit, use_bias=use_bias, activation=tf.sigmoid,
                                     name="trans_gate")
        hidden = tf.layers.dense(inputs, units=num_unit, use_bias=use_bias, activation=activation, name="hidden")
        carry_gate = tf.subtract(1.0, trans_gate, name="carry_gate")
        output = tf.add(tf.multiply(hidden, trans_gate), tf.multiply(inputs, carry_gate), name="output")
        return output


def char_cnn_hw(inputs, kernel_sizes, filters, dim, hw_layers, padding="VALID", activation=tf.nn.relu, use_bias=True,
                hw_activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="char_cnn_hw"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        outputs = []
        for i, (kernel_size, filter_size) in enumerate(zip(kernel_sizes, filters)):
            weight = tf.get_variable("filter_%d" % i, shape=[1, kernel_size, dim, filter_size], dtype=tf.float32)
            bias = tf.get_variable("bias_%d" % i, shape=[filter_size], dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding=padding, name="conv_%d" % i) + bias
            pool = tf.reduce_max(activation(conv), axis=2)
            outputs.append(pool)
        outputs = tf.concat(values=outputs, axis=-1)
        for i in range(hw_layers):
            outputs = highway_layer(outputs, num_unit=sum(filters), activation=hw_activation, use_bias=use_bias,
                                    reuse=reuse, name="highway_%d" % i)
        return outputs


def bidirectional_rnn(inputs, seq_len, training, num_units, drop_rate=0.0, activation=tf.tanh, concat=True,
                      reuse=tf.AUTO_REUSE, name="bi_rnn"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units, name="forward_lstm_cell")
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units, name="backward_lstm_cell")
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
        if concat:
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.layers.dropout(outputs, rate=drop_rate, training=training)
            outputs = activation(outputs)
        else:
            output1 = tf.layers.dense(outputs[0], units=num_units, use_bias=True, reuse=reuse, name="forward_dense")
            output1 = tf.layers.dropout(output1, rate=drop_rate, training=training)
            output2 = tf.layers.dense(outputs[1], units=num_units, use_bias=True, reuse=reuse, name="backward_dense")
            output2 = tf.layers.dropout(output2, rate=drop_rate, training=training)
            bias = tf.get_variable(name="bias", shape=[num_units], dtype=tf.float32, trainable=True)
            outputs = activation(tf.nn.bias_add(output1 + output2, bias=bias))
        return outputs


def birnn_model(inputs, seq_len, num_layers, num_units, training=False, drop_rate=0.0, activation=tf.tanh, concat=True,
                res_connect=True, reuse=tf.AUTO_REUSE, name="bi_rnn_model"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        outputs = inputs
        for layer in range(num_layers):
            output = bidirectional_rnn(outputs, seq_len, training, num_units, drop_rate, activation, concat=concat,
                                       reuse=reuse, name="bi_rnn_{}".format(layer))
            outputs = tf.add(output, outputs) if res_connect else output
        pool, _ = self_attention(outputs, project=True, reuse=tf.AUTO_REUSE, name="self_attention")
        return pool


def self_attention(inputs, project=True, reuse=tf.AUTO_REUSE, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        dim = inputs.shape[-1].value
        x = tf.layers.dense(inputs, units=dim, use_bias=False, activation=tf.tanh) if project else inputs
        weight = tf.get_variable(name="weight", shape=[dim, 1], dtype=tf.float32)
        x = tf.tensordot(x, weight, axes=1)
        alphas = tf.nn.softmax(x, axis=-1)
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)
        output = tf.squeeze(output, axis=-1)
        return output, alphas


def word_lookup(word_ids, word_dim, initial_vectors, reuse=tf.AUTO_REUSE, name="word_embedding"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        word_table = tf.Variable(initial_value=initial_vectors, name="word_table", dtype=tf.float32, trainable=False)
        unk = tf.get_variable(name="unk", shape=[1, word_dim], dtype=tf.float32, trainable=True)
        word_lookup_table = tf.concat([tf.zeros([1, word_dim]), unk, word_table], axis=0)
        word_emb = tf.nn.embedding_lookup(word_lookup_table, word_ids)
        return word_emb


def char_lookup(char_ids, char_size, char_dim, kernel_sizes, filters, reuse=tf.AUTO_REUSE, name="char_embedding"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        char_table = tf.get_variable(name="char_table", shape=[char_size - 1, char_dim], dtype=tf.float32,
                                     trainable=True)
        char_lookup_table = tf.concat([tf.zeros([1, char_dim]), char_table], axis=0)
        char_emb = tf.nn.embedding_lookup(char_lookup_table, char_ids)
        char_cnn = char_cnn_hw(char_emb, kernel_sizes, filters, char_dim, hw_layers=2, padding="VALID",
                               activation=tf.tanh, use_bias=True, hw_activation=tf.tanh, reuse=tf.AUTO_REUSE,
                               name="char_cnn_hw")
        return char_cnn


def fc_layer(features, labels, num_class, reuse=tf.AUTO_REUSE, name="output_layer"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        logits = tf.layers.dense(features, units=num_class, use_bias=True, reuse=False, name="dense")
        pred_labels = tf.argmax(logits, axis=-1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return pred_labels, loss


class TextModel:
    def __init__(self, config, num_class, num_classifier, nary_ecoc, word_dict, char_dict, vectors, ckpt_path,
                 pretrained_model=None):
        tf.reset_default_graph()
        self.cfg, self.ckpt_path, self.num_class = config, ckpt_path, num_class
        self.num_classifier, self.nary_ecoc = num_classifier, nary_ecoc
        self.word_dict, self.char_dict = word_dict, char_dict
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.ckpt_path + "log.txt")
        with tf.Graph().as_default():
            self._build_model(vectors)
            self.logger.info("total params: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                               tf.trainable_variables()])))
            self._init_session()
            if pretrained_model is not None:
                self._restore_part_weights(pretrained_model)

    def _init_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

    def _restore_part_weights(self, model_path):
        variables = slim.get_variables_to_restore()
        variables_to_restore = []
        for v in variables:
            if "word_embedding" in v.name:
                variables_to_restore.append(v)
            elif "char_embedding" in v.name:
                variables_to_restore.append(v)
            elif "embedding_projection" in v.name:
                variables_to_restore.append(v)
            elif "bi_rnn_model" in v.name:
                variables_to_restore.append(v)
            else:
                continue
        tf.train.Saver(variables_to_restore).restore(self.sess, model_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.ckpt_path + "text_model", global_step=epoch)

    def restore_last_session(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No pre-trained model at {}".format(self.ckpt_path))

    def close_session(self):
        self.sess.close()

    def get_feed_dict(self, batch_words, seq_len, batch_chars, char_seq_len, labels, lr=None, training=False):
        feed_dict = {self.word_ids: batch_words, self.seq_len: seq_len, self.char_ids: batch_chars,
                     self.char_seq_len: char_seq_len}
        for i in range(self.num_classifier):
            feed_dict[self.labels[i]] = labels[i]
        if lr is not None:
            feed_dict[self.lr] = lr
        feed_dict[self.training] = training
        return feed_dict

    def _build_model(self, vectors):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        self.labels = []
        for i in range(self.num_classifier):
            self.labels.append(tf.placeholder(tf.int32, shape=[None, self.num_class], name="labels_%d" % i))
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.training = tf.placeholder(tf.bool, shape=[], name="training")

        # inputs
        word_emb = word_lookup(self.word_ids, self.cfg.word_dim, vectors)
        char_emb = char_lookup(self.char_ids, len(self.char_dict), self.cfg.char_dim, self.cfg.kernel_sizes,
                               self.cfg.filters)
        emb = tf.concat([word_emb, char_emb], axis=-1, name="emb")
        emb = tf.layers.dense(emb, units=self.cfg.num_units, name="emb_projection")

        # encoder
        features = birnn_model(emb, self.seq_len, self.cfg.num_layers, self.cfg.num_units, training=self.training,
                               drop_rate=self.cfg.drop_rate, activation=tf.tanh, concat=False, res_connect=True)

        # decoders
        self.pred_labels, self.loss = [], []
        for i in range(self.num_classifier):
            pred_labels, loss = fc_layer(features, self.labels[i], self.num_class, name="out_%d" % i)
            pred_labels = tf.expand_dims(pred_labels, axis=-1)
            self.pred_labels.append(pred_labels)
            self.loss.append(loss)

        self.pred_labels = tf.concat(self.pred_labels, axis=-1)
        self.loss = tf.add_n(self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.cfg.grad_clip is not None:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def train(self, train_words, train_chars, train_labels, test_words, test_chars, test_labels):
        global_test_acc, global_step, lr = 0.0, 0, self.cfg.lr
        num_batches = math.ceil(float(len(train_words) / self.cfg.batch_size))

        self.logger.info("start training...")
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            train_words, train_chars, train_labels = sklearn.utils.shuffle(train_words, train_chars, train_labels)
            prog = Progbar(target=num_batches)

            for i, (b_words, b_seq_len, b_chars, b_char_seq_len, b_labels) in enumerate(batch_iter(
                    train_words, train_chars, train_labels, self.cfg.batch_size)):
                global_step += 1
                batch_labels = []
                for j in range(self.num_classifier):
                    ecoc_array = self.nary_ecoc[:, j]
                    b_lbs = remap_labels(b_labels.copy(), ecoc_array)
                    b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                    batch_labels.append(b_lbs)
                feed_dict = self.get_feed_dict(b_words, b_seq_len, b_chars, b_char_seq_len, batch_labels, lr=lr,
                                               training=True)
                _, pred_labels, loss = self.sess.run([self.train_op, self.pred_labels, self.loss], feed_dict=feed_dict)
                acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, b_labels)
                prog.update(i + 1, [("Global Step", global_step), ("Train Loss", loss), ("Train Acc", acc * 100)])
            accuracy, _ = self.test(test_words, test_chars, test_labels, batch_size=200, print_info=True, restore=False)

            if accuracy > global_test_acc:
                global_test_acc = accuracy
                self.save_session(epoch)
            lr = self.cfg.lr / (1 + epoch * self.cfg.lr_decay)

    def test(self, test_words, test_chars, test_labels, batch_size, print_info=True, restore=True):
        if restore:
            self.restore_last_session()
        accuracies, predictions = list(), list()
        for i, (b_words, b_seq_len, b_chars, b_char_seq_len, b_labels) in enumerate(batch_iter(
                test_words, test_chars, test_labels, batch_size)):
            batch_labels = []
            for j in range(self.num_classifier):
                ecoc_array = self.nary_ecoc[:, j]
                b_lbs = remap_labels(b_labels.copy(), ecoc_array)
                b_lbs = dense_to_one_hot(b_lbs, self.num_class)
                batch_labels.append(b_lbs)
            feed_dict = self.get_feed_dict(b_words, b_seq_len, b_chars, b_char_seq_len, batch_labels)
            pred_labels = self.sess.run(self.pred_labels, feed_dict=feed_dict)
            acc = compute_ensemble_accuracy(pred_labels, self.nary_ecoc, b_labels)
            accuracies.append(acc)
            predictions.append(pred_labels)
        accuracy = np.mean(accuracies)
        predictions = np.concatenate(predictions, axis=0)
        if print_info:
            self.logger.info(" -- Test Accuracy: {:.4f}".format(accuracy * 100))
        return accuracy, np.reshape(predictions, newshape=(predictions.shape[0], self.num_classifier))
