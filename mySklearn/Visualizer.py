import matplotlib.pyplot as plt
import numpy as np

from mySklearn import Measurer


def plot_rgr(X, y):
    plt.plot(X, y, 'bo')


def draw_digits(X, ind):
    demo = np.hstack([X[k] for k in ind])
    plt.figure(figsize=(15, 15))
    plt.imshow(demo, cmap='gray')
    plt.show()


def plot_clf(X, y, title='Data'):
    classes = np.unique(y)
    styles = ['bo', 'ro', 'go', 'co', 'mo', 'b.', 'r.', 'g.', 'c.', 'm.']
    if classes.size > len(styles):
        raise ValueError('too many classes')
    if X.ndim > 2:
        print('sorry, bro')
        return
    for i, cl in enumerate(classes):
        ind = np.where(y == cl)
        plt.plot(X[ind, 0], X[ind, 1], styles[i])
        plt.title(title)


def show_clf_res(X, y, y_pred, average='macro'):
    plt.subplot(1, 2, 1)
    plot_clf(X, y, 'True')
    plt.subplot(1, 2, 2)
    plot_clf(X, y_pred, 'Prediction')
    plt.show()

    print(
        f'F1 {average}: {Measurer.f1(y_pred, y, average=average)} \nprecisions: {Measurer.precision_(y_pred, y)}\nrecalls: {Measurer.recall_(y_pred, y)}')
