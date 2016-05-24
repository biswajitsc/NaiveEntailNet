
class Options(object):
    batch_size = 100
    max_seq_length = 100
    inp_dim = 100
    lstm_size = 100
    lstm_layers = 3
    random_init_width = 0.1
    ent_tensor_width = 100
    dataset_location = './dataset/sick_train/SICK_train.txt'
    large_word2vec_model = 'large_word2vec_model.txt'
    small_word2vec_model = 'small_word2vec_model.txt'
    sick_vocab_file = 'sick_vocab.txt'
