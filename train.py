import argparse

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from mySklearn.Classifiers import LogisticRegression, SVM
from mySklearn.Measurer import f1
from mySklearn.Preprocessor import scale, PCA
from utils import read_idx, save_model


def preprocess(X, y, bin_threshold=0.95, pca_exp_var=0.99):
    X = X.reshape((X.shape[0], -1))
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    pca = PCA(pca_exp_var)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test, pca


def parse_args():
    path_to_x_train = 'samples/train-images-idx3-ubyte.gz'
    path_to_y_train = 'samples/train-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_train_dir', default=path_to_x_train,
                        help=f'path to the file with the training sample\'s records, '
                             f'default: {path_to_x_train}')
    parser.add_argument('-y', '--y_train_dir', default=path_to_y_train,
                        help=f'path to the file with the training sample\'s labels, '
                             f'default: {path_to_y_train}')
    parser.add_argument('-m', '--model_output_dir', default=path_to_model,
                        help='path to the file for saving model, '
                             f'default: {path_to_model}')
    parser.add_argument('-v', '--verbose', default='tqdm',
                        help='is verbosity, options: True, False, tqdm, default: tqdm')
    parser.add_argument('-i', '--max_iter', type=int, default=1000,
                        help='number of iterations, default: 1000')
    parser.add_argument('-ih', '--max_iter_hyp', type=int, default=500,
                        help='number of iterations for hyper params optimizing, default: 500')

    return parser.parse_args()


def main():
    args = parse_args()

    path_to_x_train = args.x_train_dir
    path_to_y_train = args.y_train_dir
    path_to_model = args.model_output_dir
    verbose = args.verbose
    max_iter = args.max_iter
    max_iter_hyp = args.max_iter_hyp

    X = read_idx(path_to_x_train)
    y = read_idx(path_to_y_train)

    X_train, X_test, y_train, y_test, pca = preprocess(X, y)
    model_params = {
        'model': [SVM, LogisticRegression],
        'C': [1e5, 1e6, 1e7],  # [1, 10, 1e2, 1e5, 1e7, 1e10],
        'step': [50, 100, 150, 250],  # [10, 50, 100, 150, 250]
    }

    models_res = {}
    for model in model_params['model']:
        for C in model_params['C']:
            for step in model_params['step']:
                print(f'testing {model.__name__} C={C} step={step}...')
                m = model(C=C, step=step, verbose=verbose, max_iter=max_iter_hyp)
                m.fit(X_train, y_train)
                f1_score = f1(m.predict(X_test), y_test)
                print(f'         holdout f1 = {f1_score}')
                models_res.update({f1_score: {'model': model, 'C': C, 'step': step}})
    best_params = models_res[max(models_res)]
    print(f'best_params: {best_params}')
    print('fitting the best model...')

    best_model = best_params['model'](C=best_params['C'], step=best_params['step'], max_iter=max_iter, verbose=verbose)
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    best_model.fit(X, y)
    print(classification_report(y_test, best_model.predict(X_test), digits=5))

    print(f'Saving model to {path_to_model}')
    save_model(path_to_model, {'model': best_model,
                               'pca': pca})


if __name__ == "__main__":
    main()
