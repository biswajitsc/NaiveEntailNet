'''
Basic utilities to read dataset
'''

import tensorflow as tf
import utils
import numpy as np
import sys

from tensorflow.models.rnn import rnn, rnn_cell
from options import Options


def extract_last_relevant(outputs, length):
    """
    Source: http://stackoverflow.com/questions/35835989/
    how-to-pick-the-last-valid-output-values-from-tensorflow-rnn
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the
            output activations of each in the batch for each time step as
            returned by tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in
            the batch with all later time steps being zeros. Should be of type
            tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation
            for each example in the batch.
    """
    output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    # Query shape.
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    num_neurons = int(output.get_shape()[2])
    # Index into flattened array as a workaround.
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, num_neurons])
    relevant = tf.gather(flat, index)
    return relevant


def RNN(input_seq, input_len, scope_name, reuse, initial_state, keep_prob):
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
            initializer=Options.initializer()
        )
        lstm_cell_fw = rnn_cell.DropoutWrapper(
            lstm_cell_fw,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob
        )
        if Options.lstm_layers > 1:
            lstm_cell_fw = rnn_cell.MultiRNNCell(
                [lstm_cell_fw] * Options.lstm_layers
            )

        lstm_cell_bw = rnn_cell.LSTMCell(
            Options.lstm_dim,
            initializer=Options.initializer()
        )
        lstm_cell_bw = rnn_cell.DropoutWrapper(
            lstm_cell_bw,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob
        )
        if Options.lstm_layers > 1:
            lstm_cell_bw = rnn_cell.MultiRNNCell(
                [lstm_cell_bw] * Options.lstm_layers
            )

        outputs, state_fw, state_bw = rnn.bidirectional_rnn(
            cell_fw=lstm_cell_fw,
            cell_bw=lstm_cell_bw,
            inputs=X,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=tf.float32,
            sequence_length=input_len
        )

    output_bw = outputs[0]
    _, output_bw = tf.split(1, 2, output_bw)

    output_fw = extract_last_relevant(outputs, input_len)
    output_fw, _ = tf.split(1, 2, output_fw)

    output = tf.concat(1, [output_fw, output_bw])

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

        self.initial_state = tf.placeholder(
            tf.float32,
            [None, 2 * Options.lstm_dim * Options.lstm_layers],
            'lstm_init'
        )

        self.keep_prob = tf.placeholder(
            tf.float32,
            [],
            'p_keep'
        )

        self.state1 = RNN(self.input_seq1, self.input_len1,
                          'lstm', None, self.initial_state, self.keep_prob)
        self.state2 = RNN(self.input_seq2, self.input_len2,
                          'lstm', True, self.initial_state, self.keep_prob)

        W = tf.get_variable(
            'W_tensor',
            shape=[2 * Options.lstm_dim, 2 * Options.lstm_dim,
                   Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L1 = tf.get_variable(
            'L1_tensor',
            shape=[2 * Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L2 = tf.get_variable(
            'L2_tensor',
            shape=[2 * Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        C = tf.get_variable(
            'C_tensor',
            shape=[1, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        W = tf.reshape(W, [2 * Options.lstm_dim, -1])

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

        W_s = tf.get_variable(
            'W_softmax',
            shape=[Options.ent_tensor_width, Options.num_classes],
            initializer=Options.initializer()
        )

        # b_s = tf.get_variable(
        #     'b_softmax',
        #     shape=[Options.num_classes],
        #     initializer=Options.initializer()
        # )

        logits = tf.matmul(temp, W_s)
        self.pred = logits

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
                            self.keep_prob: Options.keep_prob
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

                # if i == 20:
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
                    self.keep_prob: 1.0
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

            for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                val1, val2 = sess.run([self.state1, self.state2], feed_dict={
                    self.input_seq1: d1,
                    self.input_len1: l1,
                    self.input_seq2: d2,
                    self.input_len2: l2,
                    self.labels: l,
                    self.initial_state: np.zeros((
                        Options.batch_size,
                        2 * Options.lstm_dim * Options.lstm_layers
                    )),
                    self.keep_prob: 1.0
                })

                break
