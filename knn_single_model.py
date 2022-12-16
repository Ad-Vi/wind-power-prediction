import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def weights(distances):
    return 1 / (distances + 10)

def distance(x0, x1):
    return np.linalg.norm(x0 - x1)

def f(k, X_train, X_test, Y_train, Y_test):
    k = max(int(k), 1)
    knn = KNeighborsRegressor(n_neighbors=k, weights=weights)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)
    return mean_absolute_error(Y_test, predictions)

def compute_k(X_train, X_test, Y_train, Y_test):
    min_k = 1
    max_k = 100
    best_k = dichotomic_search(f, [min_k, max_k], (X_train, X_test, Y_train, Y_test))
    
    start_k = max(best_k - 100, min_k)
    end_k = min(best_k + 100, max_k)
    k_values = np.arange(start_k, end_k, 8)
    errors = []
    for k in k_values:
        errors.append(f(k, X_train, X_test, Y_train, Y_test))

    return best_k, k_values, np.array(errors)

def dichotomic_search(func, interval, args):
    step = np.ceil((interval[1] - interval[0]) / 2)
    a = min(step + interval[0], interval[1])
    av = func(a, *args)
    while step > 1:
        step = np.ceil(step / 2)
        b = max(a - step, interval[0])
        c = min(a + step, interval[1])
        bv = func(b, *args)
        cv = func(c, *args)
        if bv < av and bv < cv:
            a = b
            av = bv
        elif cv < av:
            a = c
            av = cv
        print("Dichotomic search: ", a)
    return int(a)

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

#def mean_absolute_error(y1, y2):
#    return np.mean(np.abs(y1 - y2))


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
    Y_true_format = 'data/Y_true_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)

    # Read input and output files (1 per zone)
    Xs, Ys, Ys_true = [], [], []
    for i in range(N_ZONES):
        Xs.append(pd.read_csv(X_format.format(i=i+1)))
        Ys.append(pd.read_csv(Y_format.format(i=i+1)))
        Ys_true.append(pd.read_csv(Y_true_format.format(i=i+1)))

        # Flatten temporal dimension (NOTE: this step is not compulsory)
        if flatten:
            X_train, X_test, Y_train = split_train_test(Xs[i], Ys[i])
            Xs[i] = (X_train, X_test)
            Ys[i] = Y_train

    # Fit your models here
    features = ['U10', 'U100', 'V10', 'V100']
    date_features = ['Hour', 'Day', 'Month', 'Year']
    zone_distance = 100
    n_features = N_ZONES + len(features) + 1 # + 1 for timestamp
    n_date_features = len(date_features)
    n_samples = len(Xs[0][0]['ZONEID'])
    train_size = int(n_samples * .8)
    test_size = n_samples - train_size
    predict_size = len(Xs[0][1]['ZONEID'])

    X = np.zeros((N_ZONES*n_samples, n_features))
    Y = np.zeros(N_ZONES*n_samples)
    X_predict = np.zeros((N_ZONES*predict_size, n_features))
    Y_predict = np.zeros(N_ZONES*predict_size)

    dates = np.zeros((N_ZONES*n_samples, n_date_features))
    predict_dates = np.zeros((N_ZONES*predict_size, n_date_features))

    zones = np.zeros((N_ZONES*n_samples, N_ZONES))
    predict_zones = np.zeros((N_ZONES*predict_size, N_ZONES))

    for i in range(N_ZONES):
        Y_predict[i*predict_size:(i+1)*predict_size] = np.array(Ys_true[i]).reshape(predict_size)

        # Inputs
        for j, feature in enumerate(features):
            for k, value in enumerate(Xs[i][0][feature]):
                X[i*n_samples + k][j] = value
            for k, value in enumerate(Xs[i][1][feature]):
                X_predict[i*predict_size + k][j] = value

        # Dates
        for j, feature in enumerate(date_features):
            for k, value in enumerate(Xs[i][0][feature]):
                dates[i*n_samples + k][j] = value
            for k, value in enumerate(Xs[i][1][feature]):
                predict_dates[i*predict_size + k][j] = value

        # Zones
        for k, value in enumerate(Xs[i][0]['ZONEID']):
            zones[i*n_samples + k][int(value) - 1] = zone_distance
        for k, value in enumerate(Xs[i][1]['ZONEID']):
            predict_zones[i*predict_size + k][int(value) - 1] = zone_distance

        
        # Outputs
        for k, value in enumerate(Ys[i]['TARGETVAR']):
            Y[i*n_samples + k] = value

    # Add dates and zones to feature vectors
    date_weight = .1
    first_ts = feature_vec_to_ts(dates[0])
    last_ts = feature_vec_to_ts(dates[n_samples - 1]) # same as last element
    for i in range(n_samples*N_ZONES):
        ts = feature_vec_to_ts(dates[i])
        X[i][n_features - 1] = date_weight * (ts - first_ts) / (last_ts - first_ts)

        zone_index = len(features)
        X[i][zone_index: zone_index + N_ZONES] = zones[i]
        
    first_ts = feature_vec_to_ts(predict_dates[0])
    last_ts = feature_vec_to_ts(predict_dates[predict_size - 1])
    for i in range(predict_size*N_ZONES):
        ts = feature_vec_to_ts(predict_dates[i])
        X_predict[i][n_features - 1] = date_weight * (ts - first_ts) / (last_ts - first_ts)

        zone_index = len(features)
        X[i][zone_index: zone_index + N_ZONES] = predict_zones[i]
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=43)

    n_neighbors = 24

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 17,
        'font.serif': 'Computer Modern Roman',
        'axes.grid': True,
        'grid.alpha': .7,
        'savefig.transparent': True,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'markers.fillstyle': 'none',
        'lines.marker': 'o',
        'lines.linewidth': 2.1,
        'axes.formatter.limits' : (0, 0),
    })

    """errors = compute_k(X_train, X_test, Y_train, Y_test)
    plt.figure()
    plt.plot(errors[1], errors[2])
    print(errors[0])
    n_neighbors = int(errors[0])
    plt.xlabel(r"$k$")
    plt.ylabel("MAE")
    plt.savefig(f"figs/mae_single_model")"""

    # Test predictions
    selection = np.repeat(np.load('selection.npy'), N_ZONES)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)


    error = mean_absolute_error(Y_test, predictions)

    # Real predictions
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X, Y)
    predictions = knn.predict(X_predict)
    gradescope_mae = mean_absolute_error(Y_predict[selection == True], predictions[selection == True])
    gradescope_full_mae = mean_absolute_error(Y_predict, predictions)
    print(X_predict)
    print(Y_predict)
    print(predictions)
    print("MAE: ", error)
    print("Gradescope MAE: ", gradescope_mae)
    print("Gradescope full MAE: ", gradescope_full_mae)
    plt.show()