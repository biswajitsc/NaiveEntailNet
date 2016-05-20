import tensorflow as tf
import rnn

from rnn import rnn_cell


class Options(object):
    def __init__(self):
        self.batch_size = 100
        self.max_seq_length = 100
        self.inp_dim = 100
        self.lstm_size = 100
        self.lstm_layers = 3
        self.random_init_width = 0.1
        self.ent_tensor_width = 100


class EntailModel(object):

    def __init__(self, options):

        input_seq1 = tf.placeholder(
            tf.float32,
            [options.batch_size, options.max_seq_length, options.inp_dim],
            'sent1'
        )
        input_len1 = tf.placeholde(tf.int32, [options.batch_size], 'len1')

        input_seq2 = tf.placeholder(
            tf.float32,
            [options.batch_size, options.max_seq_length, options.inp_dim],
            'sent2'
        )
        input_len2 = tf.placeholde(tf.int32, [options.batch_size], 'len2')

        with tf.variable_scope('lstm'):
            lstm_cell = rnn_cell.BasicLSTMCell(options.lstm_size)
            lstm = rnn_cell.MultiRNNCell([lstm_cell] * options.lstm_layers)

            initial_state = lstm.zero_state(options.batch_size, tf.float32)

            lstm_output1, _ = rnn.rnn(
                cell=lstm,
                inputs=input_seq1,
                initial_state=initial_state,
                dtype=tf.float32,
                sequence_length=input_len1
            )

        with tf.variable_scope('lstm', reuse=True):
            lstm_output2, _ = rnn.rnn(
                cell=lstm,
                inputs=input_seq2,
                initial_state=initial_state,
                dtype=tf.float32,
                sequence_length=input_len2
            )

        tf.Variable(
            tf.random_uniform(
                [options.ent_tensor_width, options.lstm_size, options.lstm_size],
                -options.random_init_width, options.random_init_width
            ),
            name='W_tensor'
        )


