"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of CFX+SVM.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Codes import k_cross_validation

'''
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

#normalized data
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
INITIAL_NEURAL_ACTIVITY = [0.51]
DISCRIMINATION_THRESHOLD = np.arange(0.5, 1, 0.01)
EPSILON = np.arange(0.2, 0.5, 0.01)
k_cross_validation(FOLD_NO, X_train, y_train, X_test, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)


