import tensorflow as tf


class Options(object):
    batch_size = 100
    max_seq_length = 32
    inp_dim = 100
    num_classes = 3

    lstm_dim = 300
    lstm_layers = 1

    ent_tensor_width = 10

    dataset_location = './dataset/sick_train/SICK_train.txt'
    large_word2vec_model = 'large_glove_model.txt'
    small_word2vec_model = 'small_glove_model.txt'
    sick_vocab_file = 'sick_vocab.txt'

    learning_rate = 0.001
    momentum = 0.9
    train_iters = 20

    stddev = 0.01

    def initializer():
        return tf.random_normal_initializer(stddev=Options.stddev)
