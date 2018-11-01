import argparse

from sklearn.metrics import classification_report

from mySklearn.Preprocessor import scale
from utils import read_idx, load_model


def preprocess(X, pca, bin_threshold=0.95):
    X = X.reshape((X.shape[0], -1))
    X = scale(X)
    X = pca.transform(X)
    return X


def parse_args():
    path_to_x_test = 'samples/t10k-images-idx3-ubyte.gz'
    path_to_y_test = 'samples/t10k-labels-idx1-ubyte.gz'
    path_to_model = 'samples/my_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_test_dir', default=path_to_x_test,
                        help=f'path to the file with the testing sample\'s records, '
                             f'default: {path_to_x_test}')
    parser.add_argument('-y', '--y_test_dir', default=path_to_y_test,
                        help=f'path to the file with the testing sample\'s labels, '
                             f'default: {path_to_y_test}')
    parser.add_argument('-m', '--model_input_dir', default=path_to_model,
                        help='path to the file for loading model, '
                             f'default: {path_to_model}')

    return parser.parse_args()


def main():
    args = parse_args()

    X = read_idx(args.x_test_dir)
    y = read_idx(args.y_test_dir)

    model_data = load_model(args.model_input_dir)
    model = model_data['model']
    X = preprocess(X, model_data['pca'])

    print('Metrics on the test data:\n')
    predicted_labels = model.predict(X)
    print(classification_report(y, predicted_labels, digits=5))


if __name__ == "__main__":
    main()
