import tensorflow as tf
import utils
import numpy as np
import sys

from tensorflow.python.ops import rnn, rnn_cell
from options import Options
# from sklearn.metrics import confusion_matrix


def RNN(input_seq, input_len, scope_name, reuse,
        lstm_keep_prob, nnet_keep_prob):
    with tf.variable_scope(scope_name, reuse=reuse):
        w_in = tf.get_variable(
            "w_in",
            shape=[Options.inp_dim, Options.lstm_dim],
            initializer=Options.initializer()
        )
        b_in = tf.get_variable(
            "b_in",
            shape=[Options.lstm_dim],
            initializer=Options.initializer()
        )

        X = tf.transpose(input_seq, [1, 0, 2])
        X = tf.reshape(X, [-1, Options.inp_dim])
        X = tf.matmul(X, w_in) + b_in
        X = tf.split(0, Options.max_seq_length, X)

        lstm_cell_fw = rnn_cell.LSTMCell(
            Options.lstm_dim,
            initializer=Options.initializer(),
            state_is_tuple=True
        )
        lstm_cell_fw = rnn_cell.DropoutWrapper(
            lstm_cell_fw,
            input_keep_prob=lstm_keep_prob,
            output_keep_prob=lstm_keep_prob
        )
        if Options.lstm_layers > 1:
            lstm_cell_fw = rnn_cell.MultiRNNCell(
                [lstm_cell_fw] * Options.lstm_layers
            )

        lstm_cell_bw = rnn_cell.LSTMCell(
            Options.lstm_dim,
            initializer=Options.initializer(),
            state_is_tuple=True
        )
        lstm_cell_bw = rnn_cell.DropoutWrapper(
            lstm_cell_bw,
            input_keep_prob=lstm_keep_prob,
            output_keep_prob=lstm_keep_prob
        )
        if Options.lstm_layers > 1:
            lstm_cell_bw = rnn_cell.MultiRNNCell(
                [lstm_cell_bw] * Options.lstm_layers
            )

        outputs, state_fw, state_bw = rnn.bidirectional_rnn(
            cell_fw=lstm_cell_fw,
            cell_bw=lstm_cell_bw,
            inputs=X,
            dtype=tf.float32,
            sequence_length=input_len
        )

    state_fw = tf.pack(state_fw)
    state_fw = tf.transpose(state_fw, [1, 0, 2])
    state_fw = tf.reshape(state_fw, [-1, 2 * Options.lstm_dim])

    state_bw = tf.pack(state_bw)
    state_bw = tf.transpose(state_bw, [1, 0, 2])
    state_bw = tf.reshape(state_bw, [-1, 2 * Options.lstm_dim])

    output = tf.concat(1, [state_fw, state_bw])

    return output


class EntailModel(object):

    def __init__(self):

        self.input_seq1 = tf.placeholder(
            tf.float32,
            [None, Options.max_seq_length, Options.inp_dim],
            'sent1'
        )
        self.input_len1 = tf.placeholder(
            tf.int32, [None], 'len1'
        )

        self.input_seq2 = tf.placeholder(
            tf.float32,
            [None, Options.max_seq_length, Options.inp_dim],
            'sent2'
        )
        self.input_len2 = tf.placeholder(
            tf.int32, [None], 'len2'
        )

        self.labels = tf.placeholder(
            tf.float32, [None, Options.num_classes], 'labels'
        )

        self.lstm_keep_prob = tf.placeholder(
            tf.float32,
            [],
            'p_keep'
        )

        self.nnet_keep_prob = tf.placeholder(
            tf.float32,
            [],
            'n_keep'
        )

        self.state1 = RNN(self.input_seq1, self.input_len1, 'lstm', None,
                          self.lstm_keep_prob, self.nnet_keep_prob)
        self.state2 = RNN(self.input_seq2, self.input_len2, 'lstm', True,
                          self.lstm_keep_prob, self.nnet_keep_prob)

        W = tf.get_variable(
            'W_tensor',
            shape=[2 * Options.lstm_dim, 2 * Options.lstm_dim,
                   Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L1 = tf.get_variable(
            'L1_tensor',
            shape=[4 * Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L2 = tf.get_variable(
            'L2_tensor',
            shape=[4 * Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        C = tf.get_variable(
            'C_tensor',
            shape=[1, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        W = tf.reshape(W, [4 * Options.lstm_dim, -1])

        temp = tf.matmul(self.state1, W)
        temp = tf.reshape(
            temp, [-1, 2 * Options.lstm_dim, Options.ent_tensor_width]
        )
        temp1 = tf.reshape(self.state2, [-1, 1, 2 * Options.lstm_dim])
        temp = tf.batch_matmul(temp1, temp)
        temp = tf.reshape(temp, [-1, Options.ent_tensor_width])
        temp = temp + tf.matmul(self.state1, L1) + \
            tf.matmul(self.state2, L2) + C
        temp = tf.nn.relu(temp)
        temp = tf.nn.dropout(temp, self.nnet_keep_prob)

        W_s = tf.get_variable(
            'W_softmax',
            shape=[Options.ent_tensor_width, Options.num_classes],
            initializer=Options.initializer()
        )

        b_s = tf.get_variable(
            'b_softmax',
            shape=[Options.num_classes],
            initializer=Options.initializer()
        )

        logits = tf.matmul(temp, W_s) + b_s
        self.pred = tf.argmax(logits, 1)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels)
        self.loss = tf.reduce_mean(loss)

        self.reg_loss = tf.contrib.layers.apply_regularization(
            Options.regularizer(),
            weights_list=tf.trainable_variables()
        )

        self.tot_loss = self.loss + Options.reg_weight * self.reg_loss

        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1)),
                tf.float32
            )
        )

    def train(self, seq1, len1, seq2, len2, labels,
              tseq1, tlen1, tseq2, tlen2, tlabels):
        self.lrate = tf.placeholder(
            tf.float32,
            [],
            'lrate'
        )
        optimizer = Options.optimizer(self.lrate).minimize(self.tot_loss)

        self.init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(self.init)

            decay_factor = 1
            for i in range(Options.train_iters):
                print("Iteration {}".format(i))
                loss_val = 0
                reg_loss_val = 0
                cnt = 0

                for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                    _, tlv, rlv = sess.run(
                        [optimizer, self.loss, self.reg_loss],
                        feed_dict={
                            self.lrate: Options.learning_rate / decay_factor,
                            self.input_seq1: d1,
                            self.input_len1: l1,
                            self.input_seq2: d2,
                            self.input_len2: l2,
                            self.labels: l,
                            self.initial_state: np.zeros((
                                Options.batch_size,
                                2 * Options.lstm_dim * Options.lstm_layers
                            )),
                            self.lstm_keep_prob: Options.lstm_keep_prob,
                            self.nnet_keep_prob: Options.nnet_keep_prob
                        })
                    loss_val += tlv
                    reg_loss_val += rlv
                    cnt += 1

                loss_val /= cnt
                reg_loss_val /= cnt
                print(
                    "Classification loss = {}\t Regularization loss = {}"
                    .format(loss_val, reg_loss_val)
                )

                if i % 10 == 0:
                    saver.save(sess, "model.ckpt")
                    self.test(sess, 'Train', seq1,
                              len1, seq2, len2, labels)
                    self.test(sess, 'Test', tseq1, tlen1,
                              tseq2, tlen2, tlabels)

                    sys.stdout.flush()

                # if i == 100:
                #     decay_factor *= 10

            saver.save(sess, "model.ckpt")

    def test(self, sess, mode, seq1, len1, seq2, len2, labels):
        acc = 0
        loss = 0
        cnt = 0
        for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
            tacc, tloss, tpred = sess.run(
                [self.accuracy, self.tot_loss, self.pred],
                feed_dict={
                    self.input_seq1: d1,
                    self.input_len1: l1,
                    self.input_seq2: d2,
                    self.input_len2: l2,
                    self.labels: l,
                    self.initial_state: np.zeros((
                        Options.batch_size,
                        2 * Options.lstm_dim * Options.lstm_layers
                    )),
                    self.lstm_keep_prob: 1.0,
                    self.nnet_keep_prob: 1.0
                })
            cnt += 1

            acc += tacc
            loss += tloss

        acc /= cnt
        loss /= cnt

        print("{0} Accuracy = {1}\t {0} Loss = {2}".format(mode, acc, loss))

    def exploremodel(self, seq1, len1, seq2, len2, labels):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, "model.ckpt")

            preds = []

            for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                val1 = sess.run([self.pred], feed_dict={
                    self.input_seq1: d1,
                    self.input_len1: l1,
                    self.input_seq2: d2,
                    self.input_len2: l2,
                    self.labels: l,
                    self.initial_state: np.zeros((
                        Options.batch_size,
                        2 * Options.lstm_dim * Options.lstm_layers
                    )),
                    self.lstm_keep_prob: 1.0,
                    self.nnet_keep_prob: 1.0
                })

                preds.extend(val1[0])

            classes = np.argmax(labels[:4900], axis=1)
            cm = confusion_matrix(classes, preds)
            print(cm)
            print(np.mean(np.asarray(classes) == np.asarray(preds)))
            for row in cm:
                print(row / np.sum(row))
