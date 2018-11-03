from collections import deque

import numpy as np
from tqdm import trange


def sigmoid(X, w):
    w = w.reshape((w.size // X.shape[1], X.shape[1])).T
    return 1.0 / (1.0 + np.exp(-np.longdouble(np.dot(X, w))))


def softmax(X, w):
    w = w.reshape((w.size // X.shape[1], X.shape[1])).T
    dot_prod = np.dot(X, w)
    dot_prod[dot_prod > 700] = 700
    exponents = np.exp(dot_prod)
    exp_sum = np.sum(exponents, 1).reshape(exponents.shape[0], 1) + 1e-200
    return exponents / exp_sum


def dot(X, w):
    w = w.reshape((w.size // X.shape[1], X.shape[1])).T
    return np.dot(X, w)


def regularize(penalty, reg_lambda, weights, intercept):
    if intercept == True:
        a = -1
    else:
        a = weights.size + 1
    if penalty == 'l1':
        regularizer = reg_lambda * np.sum(np.abs(weights[:a]))
    elif penalty == 'l2':
        regularizer = reg_lambda * np.sum(weights[:a] ** 2)
    else:
        raise ValueError('Wrong penalty param (only "l1" or "l2" are supported)')
    return regularizer


def numerical_gradient(loss, objective, X, y, weights, reg_lambda, intercept, eps=1e-5, penalty='l2'):
    gradient = []
    fun0 = loss(objective(X, weights), y, regularize(penalty, reg_lambda, weights, intercept))
    for i in range(weights.shape[0]):
        weights[i] += eps
        fun1 = loss(objective(X, weights), y, regularize(penalty, reg_lambda, weights, intercept))
        partial_derivative = (fun1 - fun0) / eps
        gradient.append(partial_derivative)
        weights[i] -= eps
    return np.array(gradient)


def gd(loss, objective, X, y, reg_lambda, step='auto', penalty='l2', intercept=False,
       current_w=None,
       max_iter=500, verbose='tqdm'):
    delta_loss, iter_num = 1, 0
    prev_loss = 0
    if step == 'auto':
        step = 1
    if current_w is None:
        current_w = np.zeros(X.shape[1])
    if verbose == 'tqdm':
        iterator = trange(max_iter, desc='GD', position=0)
    else:
        iterator = np.arange(max_iter)
    for iter_num in iterator:
        grad = numerical_gradient(loss, objective, X, y, current_w, reg_lambda, intercept)

        delta_w = step * grad
        current_w = current_w - delta_w

        current_loss = loss(objective(X, current_w), y,
                            regularize(penalty, reg_lambda, current_w, intercept))
        delta_loss = prev_loss - current_loss

        if delta_loss <= 0:
            step /= 10
        else:
            if step < 256:
                step *= 2

        if verbose:
            if verbose == 'tqdm':
                iterator.set_description(
                    f'SGD loss={current_loss:.4f}, step={step:.10f}')
            else:
                print(f'iter: {iter_num: .6f}, step: {step: .5f}, loss: {current_loss: .4f}')
        if abs(delta_loss) < 1e-5 or np.max(np.abs(grad)) < 1e-4:
            if iter_num > 10:
                break
    if iter_num == max_iter - 1:
        print('did not converge')
    return current_w


def sag(loss, objective, X, y, reg_lambda, step='auto', batch_size=50, min_step=1e-7,
        penalty='l2', intercept=False, current_w=None,
        max_iter=500, verbose='tqdm', memory='auto'):
    if step == 'default':
        step = 100
    if current_w is None:
        current_w = np.zeros(X.shape[1])
    if verbose == 'tqdm':
        iterator = trange(max_iter, desc='SAG', position=0)
    else:
        iterator = np.arange(max_iter)
    if memory == 'auto':
        memory = max(10, int(max_iter // 10))
    overlooking_coef = 1.0 / memory

    prev_loss = loss(objective(X[0:memory], current_w), y[0:memory],
                     regularize(penalty, reg_lambda, current_w, intercept))
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    mean_grad = np.zeros(current_w.shape[0])
    gradients = deque([mean_grad for x in range(memory)])
    '''decreasing coef is such that the change on the last iteration should be very small'''
    '''step * dec_coef**max_iter = min_step'''
    dec_coef = (min_step / step) ** (1 / max_iter)
    for iter_num in iterator:
        '''random subsample of X, y'''
        batch_index = np.random.choice(index, batch_size)
        grad = numerical_gradient(loss, objective, X[batch_index], y[batch_index, :],
                                  current_w, reg_lambda, intercept)
        left_gr = gradients.popleft()
        mean_grad -= left_gr / memory
        gradients.append(grad)
        mean_grad += grad / memory
        delta_w = step * mean_grad
        current_w = current_w - delta_w
        batch_loss = loss(objective(X[batch_index], current_w), y[batch_index, :],
                          regularize(penalty, reg_lambda, current_w, intercept))

        current_loss = (1 - overlooking_coef) * prev_loss + overlooking_coef * batch_loss

        step *= dec_coef

        if verbose:
            if verbose == 'tqdm':
                iterator.set_description(
                    f'SAG mean_loss={current_loss:.4f}, batch_loss={batch_loss:.4f} step={step:.10f}')
            else:
                print(f'iter: {iter_num: .6f}, step: {step: .5f}, loss: {current_loss: .4f}')
        prev_loss = current_loss
    return current_w
