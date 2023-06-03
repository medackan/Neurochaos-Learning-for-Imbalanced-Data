"""
Created on Mon Dec 13 04:54:36 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Dataset: Iris
Code Description:  
In low training sample regime, 150 random trials for training with [1,9] data 
instances per class are considered. Seven algorithms namely, CFX+Decision Tree, 
CFX+Random Forest, CFX+AdaBoost, CFX+Support Vector Machine (CFX+SVM), CFX+k-Nearest 
Neighbors (CFX+KNN), ChaosNet and CFX+Gaussian Naive Bayes (CFX+GNB) are tested 
in the low training sample regime on this dataset. All hyperparameters used in 
this code are obtained after hyperparameter tuning carried out in the high training 
sample regime. The performance metric used throughout all experiments in this code 
is Macro F1-score.
    
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import feature_extractor as CFX



data = pd.read_csv("churn.csv")
DATA = np.array(data)
X = DATA[:,:-1]
y = DATA[:,-1]
y = y.astype(float)


total_label = y
total_data = X
#Normalisation - Column-wise
normalized_data = (X - np.min(X,0))/(np.max(X,0) - np.min(X,0))


# Total number of classes in the dataset
NUM_CLASS = len(np.unique(total_label))

# Total number of random trials of traininfg
NUM_TRIALS = 150

# Number of traininng samples per class
NUM_SAMPLES_PER_CLASS = 10


F1SCORE_FINAL_MATRIX = np.zeros((NUM_SAMPLES_PER_CLASS, 7))



#DECISION TREE

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        
        # Hyperparameters of ChaosFEX       
        INA = 0.01 #np.arange(0.01, 0.99,0.01)
        DT = 0.94 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.01
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)
        
        clf = DecisionTreeClassifier(min_samples_leaf = 5, random_state=42, max_depth = 5, ccp_alpha = 0.00067)
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)  
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,0] = np.mean(F1_SCORE_MAT, 0)



#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.99 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.09
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

        
        clf = RandomForestClassifier( n_estimators = 10, max_depth = 6, random_state=42)
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)  
        
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,1] = np.mean(F1_SCORE_MAT, 0)



#ADABOOST
from sklearn.ensemble import AdaBoostClassifier

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.965 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.21
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

        clf = AdaBoostClassifier(n_estimators = 100, random_state=42)
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)   
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
F1SCORE_FINAL_MATRIX[:,2] = np.mean(F1_SCORE_MAT, 0)




#SVM RADIAL
from sklearn.svm import SVC

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.5 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.22
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

        clf = SVC(C = 47, kernel='rbf', decision_function_shape='ovr', random_state = 42)
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)   
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,3] = np.mean(F1_SCORE_MAT, 0)










#KNN
from sklearn.neighbors import KNeighborsClassifier

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.92 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.06
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

        clf = KNeighborsClassifier(n_neighbors = 2)
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)   
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,4] = np.mean(F1_SCORE_MAT, 0)





#CHAOSNET
from Codes import chaosnet
import feature_extractor as CFX


# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.92 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.29
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)
        
        mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)
        F1SCORE = f1_score(y_test, Y_PRED, average='macro')
        final_f1_score.append(F1SCORE)  
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,5] = np.mean(F1_SCORE_MAT, 0)




#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

# F1-Score Matrix

F1_SCORE_MAT = np.zeros((NUM_TRIALS, NUM_SAMPLES_PER_CLASS))

for NUM_SAMPLES in range(1, NUM_SAMPLES_PER_CLASS+1):
    final_f1_score = []
    
    for num_trials in range(0, NUM_TRIALS):
        # Data instamce per each class
        totaldata_instance_per_class = []
        # Percentage of training sample per each class
        low_sample_percent = []
        # Testdata instance per each class
        testdata_instance_per_class = []
        
        # Initialization of Train data (X_train) and train label (y_train)
        X_train = np.zeros((NUM_SAMPLES*NUM_CLASS, total_data.shape[1]))
        y_train = np.zeros((NUM_SAMPLES*NUM_CLASS, 1))
        
        # Testdata storing list
        testdata_list = []
        # Testlabel storing list
        testlabel_list = []
        BEGIN = 0
        FINAL = NUM_SAMPLES
        testdata_mat_instance = 0
        ## TRAIN DATA PREPARATION
        for lab in range(0, NUM_CLASS):
            
            totaldata_instance_per_class.append(normalized_data[(total_label == lab), :].shape[0])
            low_sample_percent.append(NUM_SAMPLES/totaldata_instance_per_class[lab])
            
            label = lab*np.ones((totaldata_instance_per_class[lab], 1))
            
            traindata_per_class, testdata_per_class, trainlabel_per_class, testlabel_per_class = train_test_split(normalized_data[(total_label == lab), :], label, test_size=1 - low_sample_percent[lab], random_state=num_trials)
            
            if traindata_per_class.shape[0]!= NUM_SAMPLES:
                print("**********************************")
                print("WARNING: The number of samples per class is not equal to NUM_SAMPLES")
                print("Solution: Decrease the NUM_SAMPLES_PER_CLASS")
                print("**********************************")
            testdata_mat_instance = testdata_mat_instance  + testdata_per_class.shape[0]
            testdata_instance_per_class.append(testdata_per_class.shape[0])
            
           
            X_train[BEGIN:FINAL, :] = traindata_per_class
            y_train[BEGIN:FINAL, 0] = trainlabel_per_class[:,0]
            
            BEGIN = FINAL
            FINAL = FINAL + NUM_SAMPLES
            testdata_list.append(testdata_per_class)
            testlabel_list.append(testlabel_per_class)
        # Initialization of testdata
        X_test = np.zeros((testdata_mat_instance, total_data.shape[1]))
        # Initialization of testlabel
        y_test = np.zeros((testdata_mat_instance, 1))
        
        BEGIN = 0
        FINAL = testdata_instance_per_class[0]
        ## Testdata preparation
        for lab in range(0, NUM_CLASS):
            X_test[BEGIN:FINAL, :] = testdata_list[lab]
            y_test[BEGIN:FINAL, 0] = testlabel_list[lab][:,0]
            
            BEGIN = BEGIN+testdata_instance_per_class[lab]
            if lab == NUM_CLASS-1:
                break
            else:
                FINAL = BEGIN+testdata_instance_per_class[lab+1]
                
        # Hyperparameters of ChaosFEX       
        INA = 0.51 #np.arange(0.01, 0.99,0.01)
        DT = 0.24 #np.arange(0.01, 0.99, 0.01)
        EPSILON_1 = 0.11
        
        
        FEATURE_MATRIX_TRAIN = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
        FEATURE_MATRIX_VAL = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)            

        clf = GaussianNB()
        clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

        y_pred = clf.predict(FEATURE_MATRIX_VAL)
        
        F1SCORE = f1_score(y_test, y_pred, average='macro')
        final_f1_score.append(F1SCORE)   
        
        
        print("Number of Samples per Class = ", NUM_SAMPLES, "Trial Number = ", num_trials+1, "F1-score = ", F1SCORE)                             
    # Storing F1-score random trials of training    
    F1_SCORE_MAT[:, NUM_SAMPLES-1] = final_f1_score
        
  
F1SCORE_FINAL_MATRIX[:,6] = np.mean(F1_SCORE_MAT, 0)

PATH = os.getcwd()
RESULT_PATH = PATH + '/LOW TRAINING RESULTS/' 
    
try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

    
np.save(RESULT_PATH+"/CFX_F1_SCORES.npy", F1SCORE_FINAL_MATRIX)    
 
      














