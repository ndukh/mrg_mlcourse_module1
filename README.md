# mrg_mlcourse_module1
## Solving steps
1. PCA (by 0.99% explained variance threshold; 784 features -> 331). 
2. Search over a grid of hyperparameters.
3. Choosing between Logistic Regression (softmax) and SVM (ovr) by it results.
Solver — stochastic average gradient with some modifications.
## Results
about 0.9 macro f1 in average, specific value depends on random
## Using
**Training**
```shell
$ python train.py [-h] [-x X_TRAIN_DIR] [-y Y_TRAIN_DIR] [-m MODEL_OUTPUT_DIR]
                [-v VERBOSITY] [-k K] [-i ITER_NUM]

optional arguments:
        -h, --help            show this help message and exit
        -x X_TRAIN_DIR, --x_train_dir X_TRAIN_DIR
                        path to the file with the training sample's records,
                        default: samples/train-images-idx3-ubyte.gz
        -y Y_TRAIN_DIR, --y_train_dir Y_TRAIN_DIR
                        path to the file with the training sample's labels,
                        default: samples/train-labels-idx1-ubyte.gz
        -m MODEL_OUTPUT_DIR, --model_output_dir MODEL_OUTPUT_DIR
                        path to the file for saving model, default:
                        samples/my_model
        -v VERBOSITY, --verbosity VERBOSITY
                        is verbosity, default: 1
        -k K, --k K           k-fold parameter, default: 1
        -i ITER_NUM, --iter_num ITER_NUM
                        number of iterations, default: 3000
        -ih ITER_NUM_HYP, --max_iter_hyp
                          number of iterations for gridsearch
```

**Predict**
```shell
$ python predict.py [-h] [-x X_TEST_DIR] [-y Y_TEST_DIR] [-m MODEL_INPUT_DIR]

optional arguments:
        -h, --help            show this help message and exit
        -x X_TEST_DIR, --x_test_dir X_TEST_DIR
                        path to the file with the testing sample's records,
                        default: samples/t10k-images-idx3-ubyte.gz
        -y Y_TEST_DIR, --y_test_dir Y_TEST_DIR
                        path to the file with the testing sample's labels,
                        default: samples/t10k-labels-idx1-ubyte.gz
        -m MODEL_INPUT_DIR, --model_input_dir MODEL_INPUT_DIR
                        path to the file for loading model, default:
                        samples/my_model
```
