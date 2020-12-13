import os
import math
import numpy as np
import sklearn.utils
import tensorflow as tf
from utils.logger import get_logger, Progbar
from utils.prepro_text import batch_iter, dense_to_one_hot


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
        return outputs


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


class TextModel:
    def __init__(self, config, num_class, word_dict, char_dict, vectors, ckpt_path):
        tf.set_random_seed(12345)
        tf.reset_default_graph()
        self.cfg, self.ckpt_path, self.num_class = config, ckpt_path, num_class
        self.word_dict, self.char_dict = word_dict, char_dict
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.ckpt_path + "log.txt")
        with tf.Graph().as_default():
            self._build_model(vectors)
            self.logger.info("total params: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                               tf.trainable_variables()])))
            self._init_session()

    def _init_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

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
                     self.char_seq_len: char_seq_len, self.labels: labels}
        if lr is not None:
            feed_dict[self.lr] = lr
        feed_dict[self.training] = training
        return feed_dict

    def _build_model(self, vectors):
        with tf.variable_scope("placeholders"):
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
            self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
            self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
            self.labels = tf.placeholder(tf.int32, shape=[None, self.num_class], name="labels")
            self.lr = tf.placeholder(tf.float32, name="learning_rate")
            self.training = tf.placeholder(tf.bool, shape=[], name="training")

        with tf.variable_scope("word_embedding"):
            word_table = tf.Variable(initial_value=vectors, name="word_table", dtype=tf.float32, trainable=False)
            unk = tf.get_variable(name="unk", shape=[1, self.cfg.word_dim], dtype=tf.float32, trainable=True)
            word_lookup_table = tf.concat([tf.zeros([1, self.cfg.word_dim]), unk, word_table], axis=0)
            word_emb = tf.nn.embedding_lookup(word_lookup_table, self.word_ids)

        with tf.variable_scope("char_embedding"):
            char_table = tf.get_variable(name="char_table", shape=[len(self.char_dict) - 1, self.cfg.char_dim],
                                         dtype=tf.float32, trainable=True)
            char_lookup_table = tf.concat([tf.zeros([1, self.cfg.char_dim]), char_table], axis=0)
            char_emb = tf.nn.embedding_lookup(char_lookup_table, self.char_ids)

            char_cnn = char_cnn_hw(char_emb, self.cfg.kernel_sizes, self.cfg.filters, self.cfg.char_dim, hw_layers=2,
                                   padding="VALID", activation=tf.tanh, use_bias=True, hw_activation=tf.tanh,
                                   reuse=tf.AUTO_REUSE, name="char_cnn_hw")

        with tf.variable_scope("concat_word_char_emb"):
            emb = tf.concat([word_emb, char_cnn], axis=-1)

        with tf.variable_scope("embedding_projection"):
            emb = tf.layers.dense(emb, units=self.cfg.num_units, use_bias=True, activation=None, name="emb_proj")

        outputs = birnn_model(emb, self.seq_len, self.cfg.num_layers, self.cfg.num_units, training=self.training,
                              drop_rate=self.cfg.drop_rate, activation=tf.tanh, concat=False, res_connect=True,
                              reuse=tf.AUTO_REUSE, name="bi_rnn_model")

        with tf.variable_scope("pooling_layer"):
            pool, _ = self_attention(outputs, project=True, reuse=tf.AUTO_REUSE, name="self_attention")

        with tf.variable_scope("projection"):
            self.logits = tf.layers.dense(pool, units=self.num_class, use_bias=True, reuse=False, name="dense_2")
            self.pred_labels = tf.argmax(self.logits, axis=-1)

        with tf.variable_scope("loss_and_accuracy"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                                  labels=self.labels))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels, tf.argmax(self.labels, axis=-1)),
                                                   dtype=tf.float32))

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
        train_labels = dense_to_one_hot(train_labels, self.num_class)
        self.logger.info("start training...")
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            train_words, train_chars, train_labels = sklearn.utils.shuffle(train_words, train_chars, train_labels)
            prog = Progbar(target=num_batches)
            for i, (b_words, b_seq_len, b_chars, b_char_seq_len, b_labels) in enumerate(batch_iter(
                    train_words, train_chars, train_labels, self.cfg.batch_size)):
                global_step += 1
                feed_dict = self.get_feed_dict(b_words, b_seq_len, b_chars, b_char_seq_len, b_labels, lr=lr,
                                               training=True)
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
                prog.update(i + 1, [("Global Step", global_step), ("Train Loss", loss), ("Train Acc", acc * 100)])
            accuracy, _ = self.test(test_words, test_chars, test_labels, batch_size=200, print_info=True, restore=False)
            if accuracy > global_test_acc:
                global_test_acc = accuracy
                self.save_session(epoch)
            lr = self.cfg.lr / (1 + epoch * self.cfg.lr_decay)

    def test(self, test_words, test_chars, test_labels, batch_size, print_info=True, restore=True):
        if restore:
            self.restore_last_session()
        test_labels = dense_to_one_hot(test_labels, self.num_class)
        accuracies, losses, predictions = list(), list(), list()
        for i, (b_words, b_seq_len, b_chars, b_char_seq_len, b_labels) in enumerate(batch_iter(
                test_words, test_chars, test_labels, batch_size)):
            feed_dict = self.get_feed_dict(b_words, b_seq_len, b_chars, b_char_seq_len, b_labels)
            pred_labels, loss, acc = self.sess.run([self.pred_labels, self.loss, self.accuracy], feed_dict=feed_dict)
            accuracies.append(acc)
            losses.append(loss)
            predictions.append(pred_labels)
        accuracy = np.mean(accuracies)
        loss = np.mean(losses)
        predictions = np.concatenate(predictions, axis=0)
        if print_info:
            self.logger.info(" -- Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss, accuracy * 100))
        return accuracy, np.reshape(predictions, newshape=(predictions.shape[0], 1))
