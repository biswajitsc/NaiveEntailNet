import tensorflow as tf


class Options(object):
    batch_size = 100
    max_seq_length = 40
    inp_dim = 300
    num_classes = 3

    lstm_dim = 50
    lstm_layers = 1

    ent_tensor_width = 10

    train_dataset_location = './dataset/sick_train/SICK_train.txt'
    test_dataset_location = './dataset/sick_test/SICK_test.txt'
    dataset_location_template = './dataset/sick_{0}/SICK_{0}.txt'

    word2vec_model = 'word2vec_large.txt'
    sick_vocab_file = 'sick_vocab.txt'

    learning_rate = 0.001
    momentum = 0.9
    train_iters = 1000

    stddev = 0.01

    reg_weight = 0.0
    keep_prob = 0.7

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
