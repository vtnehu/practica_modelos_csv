import platform; print(platform.platform())
import sys; print("Python", sys.version)  
import numpy; print("NumPy", numpy.__version__)  
import scipy; print("SciPy", scipy.__version__)  

import pandas as pd  
from joblib import load
from sklearn import preprocessing
import time

def inference():
    time.sleep(5)

    # Load, read and normalize training data
    testing = "./datos/test.csv"
    df_test = pd.read_csv(testing, delimiter=',')

    grouped_df_test = df_test.groupby('Year_of_Release').size().reset_index(name='conteo')
        
    y_test = grouped_df_test['conteo'].values
    X_test = grouped_df_test['Year_of_Release'].values
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    scaler = preprocessing.StandardScaler()
    X_test = grouped_df_test[['Year_of_Release', 'conteo']].values
    scaler = preprocessing.StandardScaler()
    X_test = scaler.fit_transform(X_test.reshape(-1, 2))

    
    # Models training
    
    # Run model
    clf_lda = load('./models/Inference_lda.joblib')
    print("LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    clf_nn = load('./models/Inference_NN.joblib')
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))
    
if __name__ == '__main__':
    inference()