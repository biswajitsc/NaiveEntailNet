import tensorflow as tf
import utils

from tensorflow.models.rnn import rnn, rnn_cell
from options import Options


def RNN(input_seq, input_len, scope_name, reuse):
    with tf.variable_scope(scope_name, reuse=reuse):
        w_in = tf.get_variable(
            "w_in",
            shape=[Options.inp_dim, Options.lstm_dim],
            initializer=tf.random_normal_initializer()
        )
        b_in = tf.get_variable(
            "b_in",
            shape=[Options.lstm_dim],
            initializer=tf.random_normal_initializer()
        )

        X = tf.transpose(input_seq, [1, 0, 2])
        X = tf.reshape(X, [-1, Options.inp_dim])
        X = tf.matmul(X, w_in) + b_in
        X = tf.split(0, Options.max_seq_length, X)

        lstm_cell = rnn_cell.BasicLSTMCell(Options.lstm_dim)
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * Options.lstm_layers)

        # initial_state = lstm.zero_state(None, tf.float32)

        outputs, state = rnn.rnn(
            cell=lstm,
            inputs=X,
            # initial_state=initial_state,
            dtype=tf.float32,
            sequence_length=input_len
        )

    return outputs[-1]


class EntailModel(object):

    def __init__(self):

        self.input_seq1 = tf.placeholder(
            tf.float32,
            [None, Options.max_seq_length, Options.inp_dim],
            'sent1'
        )
        self.input_len1 = tf.placeholder(
            tf.int32, [Options.batch_size], 'len1')

        self.input_seq2 = tf.placeholder(
            tf.float32,
            [None, Options.max_seq_length, Options.inp_dim],
            'sent2'
        )
        self.input_len2 = tf.placeholder(
            tf.int32, [None], 'len2')

        self.labels = tf.placeholder(tf.int32, [Options.batch_size], 'labels')

        state1 = RNN(self.input_seq1, self.input_len1, 'lstm', None)
        state2 = RNN(self.input_seq2, self.input_len2, 'lstm', True)

        W = tf.Variable(
            tf.random_normal(
                [Options.lstm_dim, Options.ent_tensor_width, Options.lstm_dim]
            ),
            name='W_tensor'
        )

        L1 = tf.Variable(
            tf.random_normal(
                [Options.lstm_dim, Options.ent_tensor_width]
            ),
            name='L1_tensor'
        )

        L2 = tf.Variable(
            tf.random_normal(
                [Options.lstm_dim, Options.ent_tensor_width]
            ),
            name='L2_tensor'
        )

        C = tf.Variable(
            tf.random_normal(
                [Options.ent_tensor_width]
            ),
            name='C_tensor'
        )

        W = tf.reshape(W, [Options.lstm_dim, -1])

        temp = tf.matmul(state1, W)
        temp = tf.reshape(temp, [-1, Options.ent_tensor_width, Options.lstm_dim])
        temp = tf.transpose(temp, [2, 0, 1])
        temp = tf.reshape(temp, [Options.lstm_dim, -1])
        print(temp)
        temp = tf.matmul(state2, W)
        temp = tf.reshape(temp, [-1, Options.ent_tensor_width])

        temp = temp + tf.matmul(state1, L1) + tf.matmul(state2, L2) + C
        temp = tf.nn.relu(temp)

        W_s = tf.Variable(
            tf.random_normal(
                [Options.ent_tensor_width, Options.num_classes]
            ),
            name='W_softmax'
        )

        b_s = tf.Variable(
            tf.random_normal(
                [Options.num_classes]
            ),
            name='b_softmax'
        )

        logits = tf.matmul(temp, W_s) + b_s

        loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels)
        self.loss = tf.reduce_mean(loss)

        self.accuracy = tf.reduce_mean(
            tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        )

        self.init = tf.initialize_all_variables()

    def train(self, seq1, len1, seq2, len2, labels):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=Options.learning_rate).minimize(self.loss)

        with tf.Session as sess:
            sess.run(self.init)
            for i in range(Options.train_iters):
                print("Iteration {}".format(i))
                for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                    sess.run(optimizer, feed_dict={
                        self.input_seq1: d1,
                        self.input_len1: l1,
                        self.input_seq2: d2,
                        self.input_len2: l2,
                        self.labels: l
                    })

                if i % 10 == 0:
                    acc = 0
                    loss = 0
                    cnt = 0
                    for d1, l1, d2, l2, l in utils.batch_iter(seq1, len1, seq2, len2, labels):
                        tacc, tloss = sess.run([self.accuracy, self.loss], feed_dict={
                            self.input_seq1: d1,
                            self.input_len1: l1,
                            self.input_seq2: d2,
                            self.input_len2: l2,
                            self.labels: l
                        })
                        cnt += 1

                        acc += tacc
                        loss += tloss

                    acc /= cnt

                    print("Accuracy = {}\tLoss = {}".format(acc, loss))
