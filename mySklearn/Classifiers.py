import numpy as np

from mySklearn import Measurer
from mySklearn import Minimizer
from mySklearn import Preprocessor


class LogisticRegression:
    def __init__(self, penalty='l2', C=1e3, max_iter=500, shuffle=True, fit_intercept=True, multi_class='softmax',
                 balancing='over', verbose='tqdm', optimizer='sag', step='auto'):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        if multi_class == 'softmax':
            self.objective = Minimizer.softmax
            self.loss = Measurer.logloss_softmax
        elif multi_class == 'ovr':
            self.objective = Minimizer.sigmoid
            self.loss = Measurer.logloss
        else:
            raise ValueError('Wrong multi_class param: only "ovr" or "softmax" are supported')
        self.multi_class = multi_class
        self.balancing = balancing
        self.verbose = verbose
        if optimizer == 'gd':
            self.optimizer = Minimizer.gd
        elif optimizer == 'sag':
            self.optimizer = Minimizer.sag
        else:
            raise ValueError('Wrong optimizer param: only "gd" or "sag" are supported')
        self.step = step
        self.coefs_ = None
        self.labels = None

    def fit(self, X, y):
        self.coefs_ = None
        if self.fit_intercept:
            X = Preprocessor.add_bias(X)
        if self.balancing:
            X, y = Preprocessor.balance_classes(X, y, self.balancing)
        if self.shuffle:
            index = np.arange(X.shape[0])
            np.random.shuffle(index, )
            X = X[index]
            y = y[index]
        labels = np.unique(y)
        y = np.vstack([Preprocessor.one_vs_all(y, label) for label in labels]).T
        w_size = X.shape[1] * y.shape[1]
        # warm_start
        if self.coefs_ is not None and (self.coefs_.size == w_size):
            w_init = self.coefs_
        else:
            w_init = np.zeros(w_size)
        self.coefs_ = self.__optimize(X, y, w_init)

    def __optimize(self, X, y, w):
        w = self.optimizer(self.loss, self.objective, X, y, reg_lambda=1.0 / self.C,
                           penalty=self.penalty, current_w=w, step=self.step,
                           intercept=self.fit_intercept, max_iter=self.max_iter, verbose=self.verbose)
        return w

    def predict_proba(self, X):
        if self.fit_intercept:
            X = Preprocessor.add_bias(X)
        if self.multi_class == 'softmax':
            probas = self.objective(X, self.coefs_)
        else:
            probas = np.hstack([self.objective(X, w) for w in self.coefs_])
        return probas

    def predict(self, X):
        y_pred = np.argmax(self.predict_proba(X), axis=1)

        return y_pred


class SVM:
    def __init__(self, C=1e3, max_iter=500, shuffle=True, fit_intercept=True, multi_class='ovr',
                 balancing='over', verbose='tqdm', optimizer='sag', step='auto'):
        self.C = C
        self.max_iter = max_iter
        self.shuffle = shuffle
        if multi_class == 'ovr':
            self.loss = Measurer.hingeloss
            self.objective = Minimizer.dot
        else:
            raise ValueError(f'{multi_class} is not implemented, only "ovr"')
        self.multi_class = multi_class
        self.balancing = balancing
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.step = step
        if optimizer == 'gd':
            self.optimizer = Minimizer.gd
        elif optimizer == 'sag':
            self.optimizer = Minimizer.sag
        else:
            raise ValueError('Wrong optimizer param: only "gd" or "sag" are supported')
        self.coefs_ = None
        self.labels = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = Preprocessor.add_bias(X)
        if self.balancing:
            X, y = Preprocessor.balance_classes(X, y, self.balancing)
        if self.shuffle:
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X = X[index]
            y = y[index]
        self.labels = np.unique(y)
        y = np.vstack([Preprocessor.one_vs_all(y, label, False) for label in self.labels]).T
        w_size = X.shape[1] * y.shape[1]
        # warm_start
        if self.coefs_ is not None and (self.coefs_.size == w_size):
            w_init = self.coefs_
        else:
            w_init = np.zeros(w_size)
        self.coefs_ = self.__optimize(X, y, w_init)

    def __optimize(self, X, y, w):
        w = self.optimizer(self.loss, self.objective, X, y,
                           reg_lambda=1.0 / (2 * self.C), penalty='l2',
                           current_w=w,
                           step=self.step, intercept=self.fit_intercept, max_iter=self.max_iter,
                           verbose=self.verbose)
        return w

    def predict(self, X):
        if self.fit_intercept:
            X = Preprocessor.add_bias(X)
        return np.argmax(self.objective(X, self.coefs_), axis=1)
