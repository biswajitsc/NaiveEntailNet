import tensorflow as tf

from tensorflow.models.rnn import rnn, rnn_cell
from options import Options


class EntailModel(object):

    def __init__(self):

        input_seq1 = tf.placeholder(
            tf.float32,
            [Options.batch_size, Options.max_seq_length, Options.inp_dim],
            'sent1'
        )
        input_len1 = tf.placeholde(tf.int32, [Options.batch_size], 'len1')

        input_seq2 = tf.placeholder(
            tf.float32,
            [Options.batch_size, Options.max_seq_length, Options.inp_dim],
            'sent2'
        )
        input_len2 = tf.placeholde(tf.int32, [Options.batch_size], 'len2')

        with tf.variable_scope('lstm'):
            lstm_cell = rnn_cell.BasicLSTMCell(Options.lstm_size)
            lstm = rnn_cell.MultiRNNCell([lstm_cell] * Options.lstm_layers)

            initial_state = lstm.zero_state(Options.batch_size, tf.float32)

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
                [Options.ent_tensor_width, Options.lstm_size, Options.lstm_size],
                -Options.random_init_width, Options.random_init_width
            ),
            name='W_tensor'
        )
