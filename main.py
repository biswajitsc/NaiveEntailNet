import numpy as np
import utils
import nnet

from options import Options


def main():
    data1, len1, data2, len2, labels = utils.read_dataset('train')
    tdata1, tlen1, tdata2, tlen2, tlabels = utils.read_dataset('test')

    print("Creating model ...")
    model = nnet.EntailModel()
    print("Model created")

    print("Training ...")
    model.train(data1, len1, data2, len2, labels, tdata1, tlen1, tdata2, tlen2, tlabels)
    print("Trained")

    # model.exploremodel(data1, len1, data2, len2, labels)


if __name__ == '__main__':
    main()
