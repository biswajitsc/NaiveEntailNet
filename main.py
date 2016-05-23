import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn, rnn_cell


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


# class EntailModel(object):

#     def __init__(self):

#         input_seq1 = tf.placeholder(
#             tf.float32,
#             [Options.batch_size, Options.max_seq_length, Options.inp_dim],
#             'sent1'
#         )
#         input_len1 = tf.placeholde(tf.int32, [Options.batch_size], 'len1')

#         input_seq2 = tf.placeholder(
#             tf.float32,
#             [Options.batch_size, Options.max_seq_length, Options.inp_dim],
#             'sent2'
#         )
#         input_len2 = tf.placeholde(tf.int32, [Options.batch_size], 'len2')

#         with tf.variable_scope('lstm'):
#             lstm_cell = rnn_cell.BasicLSTMCell(Options.lstm_size)
#             lstm = rnn_cell.MultiRNNCell([lstm_cell] * Options.lstm_layers)

#             initial_state = lstm.zero_state(Options.batch_size, tf.float32)

#             lstm_output1, _ = rnn.rnn(
#                 cell=lstm,
#                 inputs=input_seq1,
#                 initial_state=initial_state,
#                 dtype=tf.float32,
#                 sequence_length=input_len1
#             )

#         with tf.variable_scope('lstm', reuse=True):
#             lstm_output2, _ = rnn.rnn(
#                 cell=lstm,
#                 inputs=input_seq2,
#                 initial_state=initial_state,
#                 dtype=tf.float32,
#                 sequence_length=input_len2
#             )

#         tf.Variable(
#             tf.random_uniform(
#                 [Options.ent_tensor_width, Options.lstm_size, Options.lstm_size],
#                 -Options.random_init_width, Options.random_init_width
#             ),
#             name='W_tensor'
#         )


def class_map(classname):
    if classname == 'NEUTRAL':
        return 0
    elif classname == 'ENTAILMENT':
        return 1
    elif classname == 'CONTRADICTION':
        return 2
    else:
        raise Exception('Invalid classname: ' + classname)


def remove_punct(sent):
    words = sent.split()
    proc_words = []
    for word in words:
        chars = list(word)
        chars = [char for char in chars if char.isalnum()]
        proc_words.append(''.join(chars))

    proc_sent = ' '.join(proc_words).lower()
    return proc_sent


def read_sentences():
    data = []
    with open(Options.dataset_location, 'r') as readfile:
        readfile.readline()
        for line in readfile.readlines():
            line = line.strip()
            splits = line.split('\t')
            data.append((
                remove_punct(splits[1]),
                remove_punct(splits[2]),
                class_map(splits[4])
            ))

    return data


def get_vocab_file(data):
    counts = {}

    for (x, y, z) in data:
        words = (x + ' ' + y).split()
        for word in words:
            counts[word] = counts.get(word, 0) + 1

    with open(Options.sick_vocab_file, 'w') as fout:
        for word in counts:
            count = counts[word]
            fout.write(word + ' ' + str(count) + '\n')

    return counts


def get_word_vectors(data):
    counts = get_vocab_file(data)

    try:
        print('Checking if small word2vec file exists ...')
        with open(Options.small_word2vec_model, 'r'):
            pass
        print('File exists')
        print('Skipping large word2vec file')
    except FileNotFoundError:
        print('File does not exist')
        print('Getting vectors from large word2vec file ...')
        with open(Options.small_word2vec_model, 'w') as fout:
            with open(Options.large_word2vec_model, 'r') as fin:
                for line in fin:
                    words = line.split()
                    word = words[0]

                    if word in counts:
                        fout.write(line)

        print('Created small word2vec file')

    print('Reading small word2vec file ...')
    word2vec = {}
    with open(Options.small_word2vec_model, 'r') as fin:
        for line in fin:
            words = line.split()
            word = words[0]
            vector = [float(val) for val in words[1:]]
            word2vec[word] = np.asarray(vector)
    print('Reading completed')

    print(word2vec['sedan'])


def main():
    data = read_sentences()
    data = get_word_vectors(data)


if __name__ == '__main__':
    main()
