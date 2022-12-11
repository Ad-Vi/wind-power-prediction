import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


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

    # Fit your models here
    print(N_ZONES * len(Xs[0][0]['ZONEID']))
    features = ['U10', 'U100', 'V10', 'V100']
    n_features = len(features)
    train_size = len(Xs[0][0]['ZONEID'])
    test_size = len(Xs[0][1]['ZONEID'])
    X_train = np.zeros((N_ZONES, train_size, 9))
    X_test = np.zeros((N_ZONES, test_size, 9))
    Y_train = np.zeros((N_ZONES, train_size))

    for i in range(N_ZONES):
        for j, column in enumerate(Xs[i][0]):
            try:
                index = features.index(column)
                for k, value in enumerate(Xs[i][0][column]):
                    X_train[i][k][index] = value
            except ValueError:
                continue
            try:
                index = features.index(column)
                for k, value in enumerate(Xs[i][1][column]):
                    X_test[i][k][index] = value
            except ValueError:
                continue
        
        for k, value in enumerate(Ys[i]['TARGETVAR']):
            Y_train[i][k] = value

    
    print("yeah")

    for i in range(N_ZONES):
        print(i)
        forest = RandomForestRegressor(n_estimators=100)
        forest.fit(X_train[i], Y_train[i])
        #knn = KNeighborsRegressor()
        #knn.fit(X_train[i], Y_train[i])
        predictions = forest.predict(X_test[i])

        Y_test = pd.Series(predictions, index=range(len(Xs[i][1])), name='TARGETVAR')
        Y_test.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)

    #knn = KNeighborsRegressor(n_neighbors=5)
    #knn.fit(X_train, Y_train)
    #print(knn.predict(X_test))
    

    # Example: predict global training mean for each zone
    #means = np.zeros(N_ZONES)
    #for i in range(N_ZONES):
    #    means[i] = Ys[i]['TARGETVAR'].mean()

    # Write submission files (1 per zone). The predicted test series must
    # follow the order of X_test.
    #for i in range(N_ZONES):
        #Y_test = pd.Series(means[i], index=range(len(Xs[i][1])), name='TARGETVAR')
        #Y_test.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)
