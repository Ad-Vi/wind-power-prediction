import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import keras

outputFile = open('output.txt', 'w')
def printing(text):
    """Print the given text on terminal and write it to the output.

    Args:
        text (string): string to print and write
    """
    print(text)
    if outputFile:
        outputFile.write(str(text))

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

def expected_error(regressor, X_test, Y_test, is_neural_network=False):
    """computes the Mean Absolute Error (MAE) of the regressor on the test set

    Args:
        regressor : regressor used to predict the output
        X_test (dataframe): df of input features of the test set
        Y_test (dataframe): df of output values of the test set

    Returns:
        E: MAE of the regressor on the test set
    """
    if is_neural_network:
        # evaluate model on test data
        scores = regressor.evaluate(X_test, Y_test, verbose=0)
        return scores*100
    else:
        E = 0
        test_size = len(Y_test)
        predictions = np.zeros(test_size)
        # Compute average over all learning sets
        predictions = regressor.predict(X_test)
        E = (1/test_size) * np.sum(np.abs((Y_test - predictions))) # MAE
        return E

def construct_Neural_network(x_size, nbr_layers, activation='relu', kernel_initializer='he_uniform', loss='mean_squared_error', optimizer='adam', does_printing=False):
    """
    Constructs a neural network with the given parameters. All layer have the same size as the input layer.

    Args:
        x_size (int): size of the input layer
        nbr_layers (int): nbre of layer in the neural network
        activation (str, optional): Activation function in nodes. Defaults to 'relu'.
        kernel_initializer (str, optional): Initializer for the kernel weights matrix.. Defaults to 'he_uniform'.
        loss (str, optional): Loss function to use while compiling the model. Defaults to 'mean_squared_error'.
        optimizer (str, optional): optimizer to use while compiling the model. Defaults to 'adam'.
        does_printing (bool, optional): whether to print information on terminal. Defaults to False.

    Returns:
        model: Neural Network model constructed with the given parameters
        
    For more information on the parameters, see 
    https://keras.io/api/models/sequential/
    https://towardsdatascience.com/designing-your-neural-networks-a5e4617027ed
    """
    model = keras.models.Sequential()
    # add input layer
    model.add(keras.layers.Dense(x_size, input_dim=x_size, activation=activation, kernel_initializer=kernel_initializer))
    
    # add hidden layers
    for i in range(nbr_layers-2):
        model.add(keras.layers.Dense(x_size, activation=activation, kernel_initializer=kernel_initializer))
    
    # add regression output layer
    model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=kernel_initializer))
    
    # compile model
    model.compile(loss=loss, optimizer=optimizer)
    if does_printing:
        printing("Neural network :")
        printing("  Input layer : "+str(x_size)+" neurons")
        printing("  Hidden layers : "+str(nbr_layers-2)+" layers of "+str(x_size)+" neurons")
        printing("  Output layer : 1 neuron")
        printing("  Loss : "+str(loss))
        printing("  Optimizer : "+str(optimizer))
        printing("  Activation : "+str(activation))
        printing("  Kernel initializer : "+str(kernel_initializer))
        printing("  Model summary :")
        printing(model.summary())
    
    return model

def feature_selection(data, type='None', does_printing = False):
    """
    Return the data with feature selection applied to the data

    Args:
        data (dataframe): data to apply feature selection on
        type (string) : type of feature selection to apply (whether 'UnivariateVarianceTreshold' or 'Correlation'). Default to 'None'.
        does_printing (bool, optional): whether to print information on terminal. Default to False.

    Returns:
        data_copy: data with feature selection applied
    """
    if type == 'UnivariateVarianceTreshold':
        return variance_treshold_feature_selection(data, does_printing=does_printing)
    elif type == 'Correlation':
        return correlation_feature_extraction(data, does_printing=does_printing)
    else:
        return data

def variance_treshold_feature_selection(data, treshold=1e-6, does_printing=False):
    """
    Returns the data with features with variance below the treshold removed
    UNIVARIATE VARIANCE TRESHOLD
    
    Args:
        data (dataframe): data to apply feature selection on
        treshold (float) : Value for which a feature is removed if lower. Default to '1e-6'.
        does_printing (bool, optional): whether to print information on terminal. Default to False.

    Returns:
        data_copy: data with features for which the variance are lower than the tresold removed
    """
    data_copy = data.copy()
    variances = data.var(numeric_only = True)
    removed = []
    for i in range(len(variances)):
        if variances[i] < treshold:
            removed += [data.columns[i]]
            data_copy.drop(data_copy.columns[i], axis=1, inplace=True)
    if does_printing:
        if len(removed) > 0:
            printing("  Initial features :")
            printing(data.columns)
            printing("  Variances :")
            printing(variances)
            printing("  Selected features :")
            printing(data_copy.columns)
            printing("  Removed features :")
            printing(removed)
        else:
            printing("  No features removed. Features :")
            printing(data_copy.columns)
    return data_copy

def correlation_feature_extraction(data, treshold=0.9, does_printing=False):
    """
    Returns the data with features with correlation with another feature above the treshold removed
    Only one feature is removed when the correlation between two features is above the treshold.
    
    Args:
        data (dataframe): data to apply feature selection on
        treshold (float) : Value for which a feature is removed if above. Default to '0.9'.
        does_printing (bool, optional): whether to print information on terminal. Default to False.

    Returns:
        data_copy: data with features for which a covariance with another feature above the tresold are removed
    """
    corr = data.corr()
    remove, removed = [], []
    feature_to_printing = ''
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
            feature_to_printing += feature + ' (corr = ' + str(correlation) + ' with '+str(feature2)+'), '
    if does_printing:
        printing('Removed features (with correlation >='+str(treshold)+ ') : '+str(feature_to_printing))
        corr.to_csv("useless/correlation.csv", float_format='%.6f')
        plt.figure()
        plt.plot
        sns.heatmap(corr)
        plt.savefig('useless/correlation.png')
    return data_copy

def weights(distances):
    """
    Compute the weights of distances as 1 / (distances + 10)
    Weigth function usable in KNN Regressor

    Args:
        distances (array): array of dictances

    Returns:
        array: array of weigths
    """
    return 1 / (distances + 10)

def scale_data(data, n_features):
    """
    Scales the data to be between 0 and 1

    Args:
        data (dataframe): data to scale
        n_features (int): number of features in the data

    Returns:
        data: data scaled
    """
    max_features = np.full(n_features, -np.inf)
    min_features = np.full(n_features, np.inf)
    for feature_vec in data:
        for j, feature in enumerate(feature_vec):
            if max_features[j] < feature:
                max_features[j] = feature
            if min_features[j] > feature:
                min_features[j] = feature

    range_features = max_features - min_features
    data /= range_features
    return data

def feature_vec_to_ts(feature_vec):
    """
    Transforms a [hour, day, month, year] feature vector to a timestamp

    Args:
        feature_vec (numpy array): Array of features such as [hour, day, month, year]

    Returns:
        date: timestamp of the feature vector
    """
    year = int(feature_vec[3])
    month = int(feature_vec[2])
    day = int(feature_vec[1])
    hour = int(feature_vec[0])
    date = datetime(year, month, day, hour)
    return datetime.timestamp(date)

if __name__ == '__main__':
    """Main function. It performs (or not) the following steps:
    Parse arguments
    Load the data
    Perform or not feature extraction
    Perform or not data scaling
    Split the data
    Train the model
    Predict the values
    Calculate the error
    Write the predictions in submission files
    """
    # Parse arguments
    parser = ArgumentParser()
    # printing : whether information are printinged during the computing or not
    parser.add_argument('-d', '--display', type=lambda x: (str(x).lower() == 'true'), default=False, help='printing or not information while computing. Default : False')
    # Test size : proportion of the data used to test our model. By default nul.
    parser.add_argument('-t', '--test_size', type=float, default=0, help='proportion of the data used for test. Default : 0')
    # Feature selection : Feature selection to apply. None is applied if None. By default None.
    parser.add_argument('-fs', '--feature_selection', type=str, default='None', help='Feature Selection to apply : None, UnivariateVarianceTreshold, Correlation. Default : None')

    mode = "full learn"
    # Argument values
    args = parser.parse_args()
    does_printing = args.display
    test_size = args.test_size
    if test_size != 0:
        test_size = args.test_size
        mode = "learn and test"
    feature_selection = args.feature_selection
    
    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)

    # Read input and output files (1 per zone)
    # Xs : input values ; Ys : output values ; Ts : testing values ; Ms : Moments (E)
    Xs, Ys, Ts, Ms = [], [], [], []
    X_train_all = pd.DataFrame()
    Y_train_all = pd.DataFrame()
    for i in range(N_ZONES):
        if does_printing:
            printing("--Read files for zone "+str(i)+"--")
        Xs.append(feature_selection(pd.read_csv(X_format.format(i=i+1)), does_printing=does_printing))
        Ys.append(feature_selection(pd.read_csv(Y_format.format(i=i+1)), does_printing=does_printing))
        Ts.append(None)
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
                Y_train = Y_train_test[0:learn_size]
                Y_test = Y_train_test[learn_size: n_samples]
                X_test = X_train_test[learn_size: n_samples]
            # create scaler and fit it on learning data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X_train)
            # transform training, predict and test data
            X_train_scaled = scaler.transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_predict = pd.DataFrame(X_predict_scaled, columns=X_predict.columns, index=X_predict.index)
            if mode == "learn and test":
                X_test_scaled = scaler.transform(X_test)
                X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                Ts[i] = (X_test, Y_test['TARGETVAR'])
            Xs[i] = (X_train, X_predict)
            Ys[i] = Y_train
            X_train_all = pd.concat([X_train_all, X_train])
            Y_train_all = pd.concat([Y_train_all, Y_train])
    if does_printing:
        printing('Read input and output files\n')
        printing('X_train : Xs[0][0].shape ='+str(Xs[0][0].shape))
        printing('Y_train : Ys[0].shape ='+str(Ys[0].shape))
        printing('X_train_all : X_train_all.shape ='+str( X_train_all.shape))
        printing('Y_train_all : Y_train_all.shape ='+str(Y_train_all.shape))
        printing('X_predict : Xs[0][1].shape '+str(Xs[0][1].shape))
        if mode == "learn and test":
            printing('X_test : Ts[0][0].shape ='+str(Ts[0][0].shape))
            printing('Y_test : Ts[0][1].shape ='+str(Ts[0][1].shape))
    # Fit your models here
    
    # Learning algorithm -------------------------------------
    start = time.time()
    if does_printing:
        printing('Neural Network - Start')
        verbose = 2
    t = time.time()
    for i in range(N_ZONES):
        regressor = construct_Neural_network(x_size=Xs[0][0].shape[1], nbr_layers=5, does_printing=does_printing)
        regressor.fit(Xs[i][0], Ys[i]['TARGETVAR'], epochs=50, batch_size=int(X_train_all.shape[0]/10), verbose=verbose)
        if does_printing:
            printing("Regressor for zone"+str(i)+" constructed and fitted | Time :"+str(time.time()-t))
        t = time.time()
        Ys[i] = (Ys[i], np.array(regressor.predict(Xs[i][1])).flatten())
        if does_printing:
            printing("Predict Zone "+str(i)+" | Time :"+str(time.time()-t))
            printing("    Ys["+str(i)+"][1].shape = "+str(Ys[i][1].shape))
        t = time.time()
    if mode == "learn and test":
        for i in range(N_ZONES):
            E = expected_error(regressor, Ts[i][0], Ts[i][1], is_neural_network=True)
            Ms.append(E)
            if does_printing:
                printing("Expected error Zone "+str(i)+" : "+str(E))
        printing("Mean Error : "+str(np.mean(Ms))+" %")
        
    if does_printing:
        printing('Neural Network - End : ' + str(time.time() - start) + ' seconds')

    # Example: predict global training mean for each zone
    means = np.zeros(N_ZONES)
    means_prediction = np.zeros(N_ZONES)
    for i in range(N_ZONES):
        means[i] = Ys[i][0]['TARGETVAR'].mean()
        means_prediction[i] = Ys[i][1].mean()
    if does_printing:
        printing('\nmeans ='+str(means))
        printing('\nmeans_prediction ='+str(means_prediction))

    # Write submission files (1 per zone). The predicted test series must
    # follow the order of X_predict.
    printing("--------------------------------------------------------------------------------")
    for i in range(N_ZONES):
        Y_predict = pd.Series(Ys[i][1], index=range(len(Xs[i][1])), name='TARGETVAR')
        Y_predict.to_csv(f'submissions/Y_pred_Zone_{i+1}.csv', index=False)
    if does_printing:
        printing('Submission files written')
