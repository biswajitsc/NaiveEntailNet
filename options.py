
class Options(object):
    batch_size = 100
    max_seq_length = 32
    inp_dim = 300
    num_classes = 3

    lstm_dim = 100
    lstm_layers = 2

    ent_tensor_width = 100

    dataset_location = './dataset/sick_train/SICK_train.txt'
    large_word2vec_model = 'large_word2vec_model.txt'
    small_word2vec_model = 'small_word2vec_model.txt'
    sick_vocab_file = 'sick_vocab.txt'

    learning_rate = 0.1
    train_iters = 100
