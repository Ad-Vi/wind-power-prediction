import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def weights(distances):
    return 1 / (distances + 10)

def distance(x0, x1):
    return np.linalg.norm(x0 - x1)

def f(k, X_train, X_test, Y_train, Y_test):
    knn = KNeighborsRegressor(n_neighbors=k, weights=weights)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)
    return mean_absolute_error(Y_test, predictions)

def compute_k(X_train, X_test, Y_train, Y_test):
    errors = []
    k_values = range(1, 1000, 20)
    for k in k_values:
        errors.append(f(k, X_train, X_test, Y_train, Y_test))

    return k_values, np.array(errors)

def close_input_variances(X, y, dist, l):
    variances = np.zeros(l)
    index = 0
    for xi in X:
        if index >= l:
            break
        tmp = close_input_variance(xi, X, y, dist)
        if tmp > 0:
            variances[index] = tmp
            index += 1
    return variances

def close_input_variance(x0, X, y, dist):
    close_y = []
    for i, xi in enumerate(X):
        if distance(x0, xi) <= dist:
            close_y.append(y[i])
    return np.var(close_y)


def scale_data(X, n_features):
    max_features = np.full(n_features, -np.inf)
    min_features = np.full(n_features, np.inf)
    for feature_vec in X:
        for j, feature in enumerate(feature_vec):
            if max_features[j] < feature:
                max_features[j] = feature
            if min_features[j] > feature:
                min_features[j] = feature

    range_features = max_features - min_features
    X /= range_features
    return X

def feature_vec_to_ts(feature_vec):
    year = int(feature_vec[3])
    month = int(feature_vec[2])
    day = int(feature_vec[1])
    hour = int(feature_vec[0])
    date = datetime(year, month, day, hour)
    return datetime.timestamp(date)

def mean_absolute_error(y1, y2):
    return np.mean(np.abs(y1 - y2))

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
    features = ['U10', 'U100', 'V10', 'V100']
    date_features = ['Hour', 'Day', 'Month', 'Year']
    n_features = len(features) + 1 # + 1 for timestamp
    n_date_features = len(date_features)
    n_samples = len(Xs[0][0]['ZONEID'])
    train_size = int(n_samples * .9)
    test_size = n_samples - train_size
    predict_size = len(Xs[0][1]['ZONEID'])

    X = np.zeros((N_ZONES, n_samples, n_features))
    Y = np.zeros((N_ZONES, n_samples))
    X_predict = np.zeros((N_ZONES, predict_size, n_features))

    dates = np.zeros((N_ZONES, n_samples, n_date_features))
    predict_dates = np.zeros((N_ZONES, predict_size, n_date_features))

    for i in range(N_ZONES):
        # Inputs
        for j, feature in enumerate(features):
            for k, value in enumerate(Xs[i][0][feature]):
                X[i][k][j] = value
            for k, value in enumerate(Xs[i][1][feature]):
                X_predict[i][k][j] = value
        # X_train[i] = scale_data(X_train[i], n_features) # worse when data scaled
        # X_test[i] = scale_data(X_test[i], n_features)

        # Dates
        for j, feature in enumerate(date_features):
            for k, value in enumerate(Xs[i][0][feature]):
                dates[i][k][j] = value
            for k, value in enumerate(Xs[i][1][feature]):
                predict_dates[i][k][j] = value
        
        # Outputs
        for k, value in enumerate(Ys[i]['TARGETVAR']):
            Y[i][k] = value

        # Plot variances
        #plt.figure()
        #plt.plot(np.sqrt(close_input_variances(X[i], Y[i], 1, 10)))
        #plt.figure()
        #plt.plot(X_train)

    #plt.show()

    # Add dates to feature vectors
    date_weight = .1
    X_train = np.zeros((N_ZONES, train_size, n_features))
    X_test = np.zeros((N_ZONES, test_size, n_features))
    Y_train = np.zeros((N_ZONES, train_size))
    Y_test = np.zeros((N_ZONES, test_size))
    for i in range(N_ZONES):
        first_ts = feature_vec_to_ts(dates[i][0])
        last_ts = feature_vec_to_ts(dates[i][train_size - 1])
        for j, feature_vec in enumerate(dates[i]):
            ts = feature_vec_to_ts(feature_vec)
            X[i][j][n_features - 1] = date_weight * (ts - first_ts) / (last_ts - first_ts)
        
        first_ts = feature_vec_to_ts(predict_dates[i][0])
        last_ts = feature_vec_to_ts(predict_dates[i][predict_size - 1])
        for j, feature_vec in enumerate(predict_dates[i]):
            ts = feature_vec_to_ts(feature_vec)
            X_predict[i][j][n_features - 1] = date_weight * (ts - first_ts) / (last_ts - first_ts)

        X_train[i], X_test[i], Y_train[i], Y_test[i] = train_test_split(X[i], Y[i], train_size=train_size, random_state=42)

    compute_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Computed using gradescope test set
    n_neighbors = [24, 23, 987, 600, 61, 50, 64, 3676, 380, 589]
    error = 0
    """for i in compute_indexes:
        print(i)
        knn = KNeighborsRegressor(n_neighbors=n_neighbors[i], weights=weights)
        knn.fit(X_train[i], Y_train[i])
        predictions = knn.predict(X_test[i])

        error += mean_absolute_error(Y_test[i], predictions)

        # Real predictions
        knn = KNeighborsRegressor(n_neighbors=n_neighbors[i], weights=weights)
        knn.fit(X[i], Y[i])
        predictions = knn.predict(X_predict[i])
        Y_predict = pd.Series(predictions, index=range(predict_size), name='TARGETVAR')
        Y_predict.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)"""

    for i in compute_indexes:
        errors = compute_k(X_train[i], X_test[i], Y_train[i], Y_test[i])
        plt.figure()
        plt.plot(errors[0], errors[1])


    plt.show()
    

    print("MAE: ", error / N_ZONES)
    # Example: predict global training mean for each zone
    #means = np.zeros(N_ZONES)
    #for i in range(N_ZONES):
    #    means[i] = Ys[i]['TARGETVAR'].mean()

    # Write submission files (1 per zone). The predicted test series must
    # follow the order of X_predict.
    #for i in range(N_ZONES):
        #Y_test = pd.Series(means[i], index=range(len(Xs[i][1])), name='TARGETVAR')
        #Y_test.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)
