import numpy as np


def mse(y_pred, y_true, regularizer=0):
    return np.mean((y_pred - y_true) ** 2) + regularizer


def mae(y_pred, y_true, regularizer=0):
    return np.mean(np.abs(y_pred - y_true)) + regularizer


def logloss(y_pred_proba, y_true, regularizer=0):
    y_pred_proba = np.float64(y_pred_proba)
    y_pred_proba[y_pred_proba == 1.0] = 0.9999
    y_pred_proba[y_pred_proba == 0.0] = 0.0001
    return -(np.mean(y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)) - regularizer)


def logloss_softmax(y_pred_probas, y_trues, regularizer=0):
    y_pred_probas = np.float64(y_pred_probas)
    #     print(y_pred_probas.shape, y_trues.shape)
    y_pred_probas[y_pred_probas == 1.0] = 0.999999
    y_pred_probas[y_pred_probas == 0.0] = 0.000001
    log = np.log(y_pred_probas)
    y_tr_log = y_trues * log
    return -(np.mean(y_tr_log) - regularizer)


def hingeloss(y_pred, y_true, regularizer=0):
    return np.mean(np.max((np.zeros(y_pred.shape), 1 - y_pred * y_true), axis=0)) + regularizer


def calc_tp_(y_pred, y_true):
    labels = np.unique([y_pred, y_true])
    tps = {}
    for label in labels:
        tp = np.sum(np.where((y_pred == label) & (y_true == label)))  # labels are sorted ascending
        tn = np.sum(np.where((y_pred != label) & (y_true != label)))
        fp = np.sum(np.where((y_pred == label) & (y_true != label)))
        fn = np.sum(np.where((y_pred != label) & (y_true == label)))
        tps.update({label: {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}})
    return tps


def precision_(y_pred, y_true):
    tps = calc_tp_(y_pred, y_true)
    precisions = {}
    for label, dct in tps.items():
        tp, tn, fp, fn = dct['tp'], dct['tn'], dct['fp'], dct['fn']
        if tp == 0 or tp + fp == 0:
            p = 1e-28
        else:
            p = tp / (tp + fp)
        precisions.update({label: p})
    return precisions


def recall_(y_pred, y_true):
    tps = calc_tp_(y_pred, y_true)
    recalls = {}
    for label, dct in tps.items():
        tp, tn, fp, fn = dct['tp'], dct['tn'], dct['fp'], dct['fn']
        if tp == 0 or tp + fn == 0:
            r = 1e-28
        else:
            r = tp / (tp + fn)
        recalls.update({label: r})
    #     tp, tn, fp, fn = calc_tp(y_pred, y_true)
    return recalls


def f1(y_pred, y_true, alpha=0.5, average='macro', n_dig=10):
    '''alpha ~ contribution of recall: alpha = 1 => f1 = recall'''
    precisions = precision_(y_pred, y_true)
    recalls = recall_(y_pred, y_true)
    f1_scores = []
    if average == 'macro':
        for label, precision in precisions.items():
            recall = recalls[label]
            f1_scores.append(precision * recall / (alpha * precision + (1.0 - alpha) * recall))
        f1_score = np.mean(f1_scores)
    elif average == 'micro':
        mean_recall = np.mean(list(recalls.values()))
        mean_precision = np.mean(list(precisions.values()))
        f1_score = mean_precision * mean_recall / (alpha * mean_precision + (1.0 - alpha) * mean_recall)
    else:
        raise ValueError(
            'Only "micro" and "macro" averaging is supported (for binary case there is no difference between them)')
    return np.round(f1_score, n_dig)


def distance(vector, X, degree=2):
    #   some sign-trick because numpy falls on odd roots of negative numbers
    dif = np.sum(X - vector, 1) ** degree
    return np.sign(dif) * np.abs(dif) ** (1 / degree)
