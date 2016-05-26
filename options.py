import tensorflow as tf


class Options(object):
    batch_size = 100
    max_seq_length = 40
    inp_dim = 100
    num_classes = 3

    lstm_dim = 100
    lstm_layers = 1

    ent_tensor_width = 10

    train_dataset_location = './dataset/sick_train/SICK_train.txt'
    test_dataset_location = './dataset/sick_test/SICK_test.txt'
    dataset_location_template = './dataset/sick_{0}/SICK_{0}.txt'

    large_word2vec_model = 'large_glove_model.txt'
    small_word2vec_model = 'small_glove_model.txt'
    sick_vocab_file = 'sick_vocab.txt'

    learning_rate = 0.001
    momentum = 0.9
    train_iters = 1000

    stddev = 0.1

    reg_weight = 0.0001
    keep_prob = 0.5

    def initializer():
        return tf.random_normal_initializer(stddev=Options.stddev)

    def regularizer():
        return tf.contrib.layers.l2_regularizer(scale=0.9)

    def optimizer(lrate):
        return tf.train.AdamOptimizer(
            learning_rate=lrate
        )

        # return tf.train.MomentumOptimizer(
        #     learning_rate=Options.learning_rate,
        #     momentum=Options.momentum
        # )
