"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of ChaosNet on the Bank Note Authentication dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Codes import k_cross_validation

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    0 (Genuine)      -     0
    1 (Forgery)      -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    bank            -   Complete Bank Note Authentication dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_______________________________________________________________________________

CFX hyperparameter description:
_______________________________________________________________________________
    INITIAL_NEURAL_ACTIVITY         -   Initial Neural Activity.
    EPSILON                         -   Noise Intensity.
    DISCRIMINATION_THRESHOLD        -   Discrimination Threshold.
    
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)
_______________________________________________________________________________


''' 

#Already normalized data
data = pd.read_csv("churn.csv")
DATA = np.array(data)
X = DATA[:,:-1]
y = DATA[:,-1]
y = y.astype(float)
y = y.reshape(len(y),1)

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)




#validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = np.arange(0.1, 0.9, 0.01)
DISCRIMINATION_THRESHOLD = np.arange(0.91,1, 0.01)
EPSILON = np.arange(0.1, 0.5, 0.01)
k_cross_validation(FOLD_NO, X_train, y_train, X_test, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)
