import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor


def split_train_test(X, Y):
    """
    Split the input dataset (X, Y) into training and testing sets, where
    training data corresponds to true (x, y) pairs and testing data corresponds
    to (x, -1) pairs.

    Arguments
    ---------
    - X: df of input features
        The input samples.
    - Y: df of output values
        The corresponding output values.

    Return
    ------
    - X_train: df of input features
        The training set of input samples.
    - Y_train: df of output values
        The training set of corresponding output values.
    - X_test: df of input features
        The testing set of input samples. The corresponding output values are
        the ones you should predict.
    """

    X_train = X[Y['TARGETVAR'] != -1]
    Y_train = Y[Y['TARGETVAR'] != -1]
    X_test = X[Y['TARGETVAR'] == -1]
    return X_train, X_test, Y_train


if __name__ == '__main__':

    flatten = True
    does_print = False
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)

    # Read input and output files (1 per zone)
    Xs, Ys = [], []
    for i in range(N_ZONES):
        Xs.append(pd.read_csv(X_format.format(i=i+1)))
        Ys.append(pd.read_csv(Y_format.format(i=i+1)))

        # Flatten temporal dimension (NOTE: this step is not compulsory)
        if flatten:
            X_train, X_test, Y_train = split_train_test(Xs[i], Ys[i])
            Xs[i] = (X_train, X_test)
            Ys[i] = Y_train
    if does_print:
        print('Read input and output files\n')
        print('Xtrain : Xs[0][0].shape =', Xs[0][0].shape)
        print('Ytrain : Ys[0].shape =', Ys[0].shape)
        print('Xtest : Xs[0][1].shape =', Xs[0][1].shape)
    # Fit your models here
    # ...
    
    # bagging KNN -------------------------------------
    if does_print:
        print('bagging KNN - Start')
    KnnRegressor = KNeighborsRegressor(n_neighbors=100)
    bagging = BaggingRegressor(KnnRegressor, n_estimators=10)
    for i in range(N_ZONES):
        bagging.fit(Xs[i][0], Ys[i]['TARGETVAR'])
        Ys[i] = (Ys[i], bagging.predict(Xs[i][1]))
    if does_print:
        print('bagging KNN - End')

    # Example: predict global training mean for each zone
    means = np.zeros(N_ZONES)
    means_prediction = np.zeros(N_ZONES)
    for i in range(N_ZONES):
        means[i] = Ys[i][0]['TARGETVAR'].mean()
        means_prediction[i] = Ys[i][1].mean()
    if does_print:
        print('\nmeans =', means)
        print('\nmeans_prediction =', means_prediction)

    # Write submission files (1 per zone). The predicted test series must
    # follow the order of X_test.
    
    for i in range(N_ZONES):
        Y_test = pd.Series(Ys[i][1], index=range(len(Xs[i][1])), name='TARGETVAR')
        Y_test.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)
    if does_print:
        print('Write submission files')
