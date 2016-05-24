import numpy as np
import utils
import nnet

from options import Options


def main():
    data = utils.read_sentences()
    data1, len1, data2, len2, labels = utils.get_word_vectors(data)

    # print(data1.shape)
    # print(data2.shape)
    # print(len1.shape)
    # print(len2.shape)

    print("Creating model")
    model = nnet.EntailModel()
    print("Created model")
    model.train(data1, len1, data2, len2, labels)


if __name__ == '__main__':
    main()
