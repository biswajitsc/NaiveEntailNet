import numpy as np

from options import Options


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
        while not chars[-1].isalnum():
            del chars[-1]
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


def convert_to_vector(sent, vector_map):
    v_dim = len(vector_map['for'])

    words = sent.split()
    sent_proc = []

    for word in words:
        sent_proc.append(vector_map.get(word, np.zeros(v_dim)))

    return np.asarray(sent_proc)


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
            counts.pop(word)
    print('Reading completed')

    print('{} words were not found in the file.'.format(len(counts)))

    data1_proc = []
    data2_proc = []
    data_label = []

    data1_len = []
    data2_len = []

    max_len = 0
    v_dim = len(word2vec['for'])

    for (x, y, z) in data:
        x = convert_to_vector(x, word2vec)
        y = convert_to_vector(y, word2vec)
        data1_proc.append(x)
        data2_proc.append(y)
        label = np.zeros(Options.num_classes)
        label[z] = 1
        data_label.append(label)
        max_len = max(max_len, max(len(x), len(y)))

    for i in range(len(data_label)):
        elem = data1_proc[i]
        c_len = len(elem)
        elem = np.append(elem, np.zeros((max_len - c_len, v_dim)), axis=0)
        data1_proc[i] = elem
        data1_len.append(c_len)

        elem = data2_proc[i]
        c_len = len(elem)
        elem = np.append(elem, np.zeros((max_len - c_len, v_dim)), axis=0)
        data2_proc[i] = elem
        data2_len.append(c_len)

    data1_proc = np.asarray(data1_proc)
    data2_proc = np.asarray(data2_proc)
    data1_len = np.asarray(data1_len)
    data2_len = np.asarray(data2_len)
    data_label = np.asarray(data_label)

    return data1_proc, data1_len, data2_proc, data2_len, data_label


def batch_iter(data1, len1, data2, len2, labels):
    n = data1.shape[0]
    num_batches = (n + Options.batch_size - 1) // Options.batch_size
    for i in range(num_batches):
        ind1 = i * Options.batch_size
        ind2 = min(n, (i + 1) * Options.batch_size)
        yield data1[ind1:ind2], len1[ind1:ind2], data2[ind1:ind2], len2[ind1:ind2], labels[ind1:ind2]
