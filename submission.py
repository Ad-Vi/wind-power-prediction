import os
import numpy as np
import pandas as pd
import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold


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
    - X_predict: df of input features
        The testing set of input samples. The corresponding output values are
        the ones you should predict.
    """

    X_train = X[Y['TARGETVAR'] != -1]
    Y_train = Y[Y['TARGETVAR'] != -1]
    X_predict = X[Y['TARGETVAR'] == -1]
    return X_train, X_predict, Y_train

def expected_error(regressor, X_test, Y_test):
    E = 0
    test_size = len(Y_test)
    predictions = np.zeros(test_size)
    # Compute average over all learning sets
    predictions = regressor.predict(X_test)
    E = np.mean((Y_test - predictions) ** 2) # mean squared error
    return E

def feature_selection(data, does_print = False):
    # return variance_treshold_feature_selection(data, does_print=does_print)
    return correlation_feature_extraction(data, does_print=does_print)
    # return data

def variance_treshold_feature_selection(data, treshold=1e-6, does_print=False):
    data_copy = data.copy()
    variances = data.var(numeric_only = True)
    removed = []
    for i in range(len(variances)):
        if variances[i] < treshold:
            removed += [data.columns[i]]
            data_copy.drop(data_copy.columns[i], axis=1, inplace=True)
    if does_print:
        if len(removed) > 0:
            print("  Initial features :")
            print(data.columns)
            print("  Variances :")
            print(variances)
            print("  Selected features :")
            print(data_copy.columns)
            print("  Removed features :")
            print(removed)
        else:
            print("  No features removed. Features :")
            print(data_copy.columns)
    return data_copy

def correlation_feature_extraction(data, treshold=0.9, does_print=False):
    corr = data.corr()
    remove, removed = [], []
    feature_to_print = ''
    data_copy = data.copy()
    for i in range(len(data_copy.columns)):
        for j in range(i+1, len(data_copy.columns)):
            if corr.iloc[i,j] >= treshold:
                if data.columns[j+1] not in remove:
                    remove += [(data.columns[j], corr.iloc[i,j], data.columns[i])]
    for r in remove:
        feature, correlation, feature2 = r
        if feature != 'TARGETVAR' and feature not in removed:
            removed +=[feature]
            data_copy.drop(feature, axis=1, inplace=True)
            feature_to_print += feature + ' (corr = ' + str(correlation) + ' with '+str(feature2)+'), '
    if does_print:
        print('Removed features (with correlation >=', treshold, ') : ', feature_to_print)
        corr.to_csv("useless/correlation.csv", float_format='%.6f')
        plt.plot
        sns.heatmap(corr)
        plt.savefig('useless/correlation.png')
    return data_copy
        

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    # Print : whether information are printed during the computing or not
    parser.add_argument('-d', '--display', type=lambda x: (str(x).lower() == 'true'), default=False, help='print or not information while computing. Default : False')
    # Test size : proportion of the data used to test our model. By default nul.
    parser.add_argument('-t', '--test_size', type=float, default=0, help='proportion of the data used for test. Default : 0')

    mode = "full learn"
    # Argument values
    args = parser.parse_args()
    does_print = args.display
    test_size = args.test_size
    if test_size != 0:
        test_size = args.test_size
        mode = "learn and test"
    
    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)

    # Read input and output files (1 per zone)
    # Xs : input values ; Ys : output values ; Ts : testing values ; Ms : Moments (E)
    Xs, Ys, Ts, Ms = [], [], [], []
    for i in range(N_ZONES):
        if does_print:
            print("--Read files for zone "+str(i)+"--")
        Xs.append(feature_selection(pd.read_csv(X_format.format(i=i+1)), does_print=does_print))
        print('Xs[i].columns ', Xs[i].columns)
        Ys.append(feature_selection(pd.read_csv(Y_format.format(i=i+1)), does_print=does_print))
        Ts.append(None)
        print('Ys[i].columns ', Ys[i].columns)
        # Flatten temporal dimension (NOTE: this step is not compulsory)
        if flatten:
            X_train_test, X_predict, Y_train_test = split_train_test(Xs[i], Ys[i])  
            if mode == "full learn":
                X_train = X_train_test
                Y_train = Y_train_test
            else:
                n_samples = X_train_test.shape[0]
                testing_size = int(0.1*n_samples)
                learn_size = n_samples - testing_size
                X_train = X_train_test[0:learn_size]
                X_test = X_train_test[learn_size: n_samples]
                Y_train = Y_train_test[0:learn_size]
                Y_test = Y_train_test[learn_size: n_samples]
                Ts[i] = (X_test, Y_test['TARGETVAR'])
            Xs[i] = (X_train, X_predict)
            Ys[i] = Y_train
    if does_print:
        print('Read input and output files\n')
        print('X_train : Xs[0][0].shape =', Xs[0][0].shape)
        print('Y_train : Ys[0].shape =', Ys[0].shape)
        print('X_predict : Xs[0][1].shape =', Xs[0][1].shape)
        if mode == "learn and test":
            print('X_test : Ts[0][0].shape =', Ts[0][0].shape)
            print('Y_test : Ts[0][1].shape =', Ts[0][1].shape)
            print('len(Y_test) = ', len(Ts[0][1]))
    # Fit your models here
    
    # Learning algorithm -------------------------------------
    start = time.time()
    if does_print:
        print('Random Forest - Start')
    regressor = RandomForestRegressor(n_estimators=100)
    for i in range(N_ZONES):
        if does_print:
            print("Fit Zone ", i)
        regressor.fit(Xs[i][0], Ys[i]['TARGETVAR'])
    for i in range(N_ZONES):
        if does_print:
            print("Predict Zone ", i)
        Ys[i] = (Ys[i], regressor.predict(Xs[i][1]))
    if mode == "learn and test":
        for i in range(N_ZONES):
            E = expected_error(regressor, Ts[i][0], Ts[i][1])
            Ms.append(E)
            if does_print:
                print("Expected error Zone ", i, " : ", E)
        print("Mean expected error : ", np.mean(Ms)*100, " %")
        
    if does_print:
        print('Random Forest - End : ' + str(time.time() - start) + ' seconds')

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
    # follow the order of X_predict.
    for i in range(N_ZONES):
        Y_predict = pd.Series(Ys[i][1], index=range(len(Xs[i][1])), name='TARGETVAR')
        Y_predict.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)
    if does_print:
        print('Write submission files')
