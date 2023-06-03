"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of CFX+Random Forest .
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import feature_extractor as CFX

'''
CFX hyperparameter description:
_______________________________________________________________________________

    INA         -   Initial Neural Activity
    EPSILON_1   -   Noise Intensity
    DT          -   Discrimination Threshold
        
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)

_______________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________

    MD      -   The maximum depth of the tree.
    NEST    -   The number of trees in the forest.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
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


#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/CFX-TUNING/RESULTS/' 
    
INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
DT = np.load(RESULT_PATH+"/h_B.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]
print('INA', INA)
print('EPSILON_1',EPSILON_1)
print('DT',DT)
print('MD',MD)
print('NEST',NEST)
FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

y_pred = clf.predict(FEATURE_MATRIX_VAL)
f1 = f1_score(y_test, y_pred, average='macro')


print('TRAINING F1 SCORE ', F1SCORE)
print('TESTING F1 SCORE ', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def plot_evaluation_metrics(algorithm_name, y_true, y_pred):
    # Convert y_true and y_pred to 1-dimensional arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,average='macro')
    rec = recall_score(y_true, y_pred,average='macro')
    f1 = f1_score(y_true, y_pred,average='macro')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.xlabel('Predicted label', fontsize=16)
    plt.ylabel('True label', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Confusion Matrix - {}'.format(algorithm_name), fontsize=16)
    plt.savefig('confusion_matrix_{}.jpeg'.format(algorithm_name))
    plt.show()
 
    
    # Print evaluation metrics
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1:', f1)

plot_evaluation_metrics('CFX_RF', y_test, y_pred)