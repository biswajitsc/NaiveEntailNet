import tensorflow as tf
import utils
import numpy as np

from tensorflow.models.rnn import rnn, rnn_cell
from options import Options


def extract_last_relevant(outputs, length):
    """
    Source: http://stackoverflow.com/questions/35835989/
    how-to-pick-the-last-valid-output-values-from-tensorflow-rnn
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
            activations of each in the batch for each time step as returned by
            tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in the
            batch with all later time steps being zeros. Should be of type tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation for
            each example in the batch.
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


def RNN(input_seq, input_len, scope_name, reuse, initial_state):
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

        lstm_cell = rnn_cell.LSTMCell(
            Options.lstm_dim,
            initializer=Options.initializer()
        )
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * Options.lstm_layers)

        outputs, state = rnn.rnn(
            cell=lstm,
            inputs=X,
            initial_state=initial_state,
            dtype=tf.float32,
            sequence_length=input_len
        )

    # outputs = tf.pack(outputs)
    # outputs = tf.transpose(outputs, [1, 2, 0])

    # temp1 = tf.range(0, Options.batch_size)
    # temp1 = tf.tile(temp1, [Options.lstm_dim])
    # temp1 = tf.reshape(temp1, [Options.lstm_dim, -1])
    # temp1 = tf.transpose(temp1, [1, 0])

    # temp2 = tf.range(0, Options.lstm_dim)
    # temp2 = tf.tile(temp2, [Options.batch_size])
    # temp2 = tf.reshape(temp2, [Options.batch_size, -1])

    # temp3 = input_len
    # temp3 = tf.tile(temp3, [Options.lstm_dim])
    # temp3 = tf.reshape(temp3, [Options.lstm_dim, -1])
    # temp3 = tf.transpose(temp3, [1, 0])

    # indices = tf.pack([temp1, temp2, temp3])
    # indices = tf.transpose(indices, [1, 2, 0])

    # outputs = tf.gather_nd(outputs, indices)

    return extract_last_relevant(outputs, input_len)


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

        self.state1 = RNN(self.input_seq1, self.input_len1,
                          'lstm', None, self.initial_state)
        self.state2 = RNN(self.input_seq2, self.input_len2,
                          'lstm', True, self.initial_state)

        W = tf.get_variable(
            'W_tensor',
            shape=[Options.lstm_dim, Options.lstm_dim,
                   Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L1 = tf.get_variable(
            'L1_tensor',
            shape=[Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        L2 = tf.get_variable(
            'L2_tensor',
            shape=[Options.lstm_dim, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        C = tf.get_variable(
            'C_tensor',
            shape=[1, Options.ent_tensor_width],
            initializer=Options.initializer()
        )

        W = tf.reshape(W, [Options.lstm_dim, -1])

        temp = tf.matmul(self.state1, W)
        temp = tf.reshape(
            temp, [-1, Options.lstm_dim, Options.ent_tensor_width]
        )
        temp1 = tf.reshape(self.state2, [-1, 1, Options.lstm_dim])
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

        logits = tf.matmul(temp, W_s)
        self.pred = logits

        loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels)
        self.loss = tf.reduce_mean(loss)

        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1)),
                tf.float32
            )
        )

    def train(self, seq1, len1, seq2, len2, labels, tseq1, tlen1, tseq2, tlen2, tlabels):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=Options.learning_rate
        ).minimize(self.loss)

        self.init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(self.init)

            for i in range(Options.train_iters):

                print("Iteration {}".format(i))
                for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                    sess.run(optimizer, feed_dict={
                        self.input_seq1: d1,
                        self.input_len1: l1,
                        self.input_seq2: d2,
                        self.input_len2: l2,
                        self.labels: l,
                        self.initial_state: np.zeros((
                            Options.batch_size,
                            2 * Options.lstm_dim * Options.lstm_layers
                        ))
                    })

                if i % 10 == 0:
                    saver.save(sess, "model.ckpt")
                    self.test(sess, 'Train', seq1, len1, seq2, len2, labels)
                    self.test(sess, 'Test', tseq1, tlen1, tseq2, tlen2, tlabels)

            saver.save(sess, "model.ckpt")

    def test(self, sess, mode, seq1, len1, seq2, len2, labels):

        acc = 0
        loss = 0
        cnt = 0
        for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
            tacc, tloss, tpred = sess.run([self.accuracy, self.loss, self.pred], feed_dict={
                self.input_seq1: d1,
                self.input_len1: l1,
                self.input_seq2: d2,
                self.input_len2: l2,
                self.labels: l,
                self.initial_state: np.zeros((
                    Options.batch_size,
                    2 * Options.lstm_dim * Options.lstm_layers
                ))
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
                val = sess.run(self.pred, feed_dict={
                    self.input_seq1: d1,
                    self.input_len1: l1,
                    self.input_seq2: d2,
                    self.input_len2: l2,
                    self.labels: l,
                    self.initial_state: np.zeros((
                        Options.batch_size,
                        2 * Options.lstm_dim * Options.lstm_layers
                    ))
                })

                print(val)

                break
