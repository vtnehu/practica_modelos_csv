import platform;
print(platform.platform())
import sys;
print("Python", sys.version)
import numpy;
print("NumPy", numpy.__version__)
import scipy;
print("SciPy", scipy.__version__)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import preprocessing



def train():
    
    training = "./datos/train.csv"
    df = pd.read_csv(training)

    grouped_df = df.groupby('Year_of_Release').size().reset_index(name='conteo')

    y_train = grouped_df['conteo'].values
    X_train = grouped_df['Year_of_Release'].values

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)

    # Data normalization (0,1)
    scaler = preprocessing.StandardScaler()
    X_train = grouped_df[['Year_of_Release', 'conteo']].values
    scaler = preprocessing.StandardScaler()
    X_train[:, 1] = scaler.fit_transform(X_train[:, 1].reshape(-1, 1)).flatten()


    # Models training

    # Linear Discrimant Analysis (Default parameters)
    
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train[:, [0, 1]], y_train)

    # Save model
    from joblib import dump
    dump(clf_lda, './models/Inference_lda.joblib')

    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(
        solver='adam',
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(500,),
        random_state=0,
        max_iter=1000
    )

    clf_NN.fit(X_train, y_train)

    # Save model
    from joblib import dump
    dump(clf_NN, './models/Inference_NN.joblib')

if __name__ == '__main__':
    train()
