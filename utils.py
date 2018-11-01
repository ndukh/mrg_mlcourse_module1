import gzip
import pickle
import struct

import numpy as np


def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_model(file, encoding='latin1'):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding=encoding)


def save_model(filename, model_data):
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
