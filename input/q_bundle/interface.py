import numpy as np


class BlackBox:

    def __init__(self):
        self._n_inputs = 3072
        self._internal_dim = 64
        self._n_outputs = 10

        self._w1 = np.zeros((self._internal_dim, self._n_inputs), dtype=np.float32)
        self._w2 = np.zeros((self._n_outputs, self._internal_dim), dtype=np.float32)
        self._x = None
        self._target = None

        filename = 'input/q_bundle/model.tsv'
        fi = open(filename, 'r')
        for i in range(self._internal_dim):
            self._w1[i, :] = np.array(fi.readline().rstrip().split('\t'), dtype=np.float32)
        self._b = np.array(fi.readline().rstrip().split('\t'), dtype=np.float32)
        for i in range(self._n_outputs):
            self._w2[i, :] = np.array(fi.readline().rstrip().split('\t'), dtype=np.float32)

        data_filename = 'input/q_bundle/images.tsv'
        fi = open(data_filename, 'r')
        splitted = fi.readline().rstrip().split('\t')
        assert len(splitted) == self._n_inputs + 1
        self._x = np.array(splitted[:-1], dtype=np.uint8)
        self._target = int(splitted[-1])

    def get_image(self):
        return self._x

    def calc(self, x):
        x = np.array(x, dtype=np.uint8)
        x = x * 0.01 - 1.28
        logits = np.dot(self._w2, np.maximum(0, np.dot(self._w1, x) + self._b))
        logits = logits - np.max(logits)
        o = np.exp(logits) / np.sum(np.exp(logits))
        return o[self._target]
