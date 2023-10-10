import sys

import matplotlib.pyplot as plt
import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_pickle(f):
    if sys.version_info[0] == 2:
        return pickle.load(f)
    elif sys.version_info[0] == 3:
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(sys.version))


def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y
