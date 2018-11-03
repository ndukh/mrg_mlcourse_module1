import numpy as np
from numpy.linalg import eig


def scale(X):
    x_max = 255
    return X / x_max


def add_bias(X):
    return np.hstack((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)))


def add_poly(X, degree):
    X_poly = X.copy()
    for d in np.arange(1, degree):
        X_poly = np.hstack((X_poly, X ** (d + 1)))
    return X_poly


def _shuffle(X, y):
    shuffled_index = np.arange(X.shape[0])
    np.random.shuffle(shuffled_index)
    return X[shuffled_index], y[shuffled_index]


def regression_preprocessing(X, y, poly_degree=1, shuffle=True):
    X = add_poly(X, poly_degree)
    X = scale(X)[0]
    X = add_bias(X)
    return train_test_split(X, y, shuffle=shuffle)


def train_test_split(X, y, train_ratio=0.65, shuffle=True):
    if shuffle:
        X, y = _shuffle(X, y)
    train_size = int(X.shape[0] * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


def balance_classes(X, y, method='oversampling'):
    labels = np.unique(y)
    class_sizes = {label: y[y == label].size for label in labels}
    if 'over' in method:
        reference = max(class_sizes.values())
    elif 'under' in method:
        reference = min(class_sizes.values())
    else:
        raise ValueError(
            f'Wrong balancing method "{method}". Only under(sampling) and over(sampling) are supported')

    for label in labels:
        positive_indices = np.array(np.where(y == label)).ravel()
        imbalance = int(reference - positive_indices.size)
        if imbalance > 0:
            to_add = np.random.choice(positive_indices, imbalance, False)
            X, y = np.vstack((X, X[to_add])), np.hstack((y, y[to_add]))
        elif imbalance < 0:
            to_del = np.random.choice(positive_indices, -imbalance, False)
            X, y = np.delete(X, to_del, 0), np.delete(y, to_del, 0)
    return X, y


def one_vs_all(y, label, zero=True):
    positive_indices = np.where(y == label)
    if zero:
        y_ = np.zeros(y.size)
    else:
        y_ = -np.ones(y.size)
    y_[positive_indices] = 1
    return y_


def dim_reduce(X):
    '''simple joining of 4 adjacent pixels into one'''
    # получилось отстойно
    n = X.shape[1]
    X = X.reshape(X.shape[0], -1, 1)
    ind = np.arange(X[0].size)
    ind = ind[:-n]
    ind = ind[(ind + 1) % n != 0]
    i = ind
    return np.mean([X[:, ind], X[:, ind + 1], X[:, ind + n], X[:, ind + 1 + n]], axis=0) \
        .reshape(X.shape[0], n - 1, -1)


def minmax3d(X):
    '''scaling 3d matrix to [0..1]'''
    xmin = np.min(X, axis=(1, 2)) * 0 + 0.01
    xmax = np.max(X, axis=(1, 2))
    X = (X - xmin[:, None, None]) / (xmax - xmin)[:, None, None]
    X[X < 0] = 0
    return X


def binarize(X, threshold=0.95):
    X[X >= threshold] = 1
    X[X < threshold] = 0
    return X


class PCA:
    def __init__(self, exp_var=None, d=None):
        self.d = d
        self.exp_var = exp_var
        if not d and not exp_var:
            raise ValueError('You should pass exp_var or d')
        self.W = None

    def fit_transform(self, X):
        X = np.float64(X)
        x_mean = np.mean(X, axis=0)
        X_centered = X - x_mean
        N = X.shape[0]
        if N > X.shape[1]:
            S = 1 / N * np.dot(X_centered.T, X_centered)
            eigvals, Q = eig(S)
            eigvals, Q = np.float64(eigvals), np.float64(Q)
            sorted_ind = eigvals.argsort()[::-1]
            if self.exp_var:
                cumsum = np.cumsum(eigvals[sorted_ind] / eigvals.sum())
                self.d = cumsum[cumsum <= self.exp_var].size + 1
            sorted_ind = sorted_ind[:self.d]
            self.W = Q[:, sorted_ind]
        else:
            raise ValueError('Not implemented for the case N <= X.shape[1]')
        T = np.dot(X_centered, self.W)
        print(f'd: {self.d}, explained var: {np.sum(eigvals[sorted_ind]) / np.sum(eigvals)}')
        return T

    def transform(self, X):
        return np.dot(X - np.mean(X, axis=0), self.W)
