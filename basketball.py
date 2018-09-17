# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 08:13:15 2018

@author: Aditya Joshi 

@Purpose: Create a classifier for a basketball shot
@Dependicies: 
    :Original data was pre-processed by converting character variables into numeric in R 
    
    :Keras used for feedforward neural network
    :scikit_learn used for generating random forest and pre-processing data,confusion matrix
    
    *Note Keras is dendent on TensorFlow and needs to be installed in Conda
"""

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return a numpy array and headers:
    """
    dataset = pd.read_csv(path,header=0, index_col=None)
    print (dataset.describe()) #used to understand the data
    header = list(dataset.columns.values)
    dataset = dataset.values
    return dataset,header


    
    

def split_2_3(dataset,header):
    """
    Split the data into two pointers and three pointers
    param dataset
    param header
    return two_pointer and three pointer numpy array
    
    *Note:This  function is not used in the main, but my intention was create a classification
    algorithm based on a 2 pointer and a 3 pointer
    """
    two_pointer = pd.DataFrame(columns=header)
    two_pointer = two_pointer.values
    three_pointer = pd.DataFrame(columns=header)
    three_pointer = three_pointer.values
    
    for i in range (0,(dataset.shape[0])):
        if (dataset[i,9] == 2):
            two_temp = dataset[i:i+1,]
            two_pointer = np.append(two_pointer,two_temp,axis = 0)
        else:
            three_temp = dataset[i:,]
            np.append(three_pointer,three_temp,axis = 0)
    

    return (two_pointer,three_pointer)        
        
        
def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset into training and testing
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :Dependent on the scikit_learn train_test_split    
    :return: train_x, test_x, train_y, test_y
    """
 

    train_x, test_x, train_y, test_y = train_test_split(dataset[:,:feature_headers],
                                                        dataset[:,target_header],
                                                        train_size=train_percentage)
    
    
    return (train_x, test_x, train_y, test_y)

def missing_values(X):
    """
    To replace missining values with an average value
    :param numpy array
    :Dependent on the imputer package in scikit_learn
    :return: imputed numpy array
    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #Making NaN values into mean
    imp = imp.fit(X)
    imp_X = imp.transform(X)
    return imp_X   
    
def random_forest_classifier(X, Y):
    """
    To train the random forest classifier with basketball parameters and target data
    :param input features(X):
    :param target features (Y):
    :Dependent on scikit_learn RandomforestClassifier    
    :return: random forest classifier
    
    """
    rf = RandomForestClassifier()        
    rf.fit(X, Y)
    return rf

def neural_network(X,Y):
    """
    To train the feedforward neural network with features and target data
    :param input features(X):
    :param target features (Y):
    :return: trained feedforward neural net algorithm
    :Dependent on the keras package
    """
    model = Sequential()
    model.add(Dense(20, input_dim=13, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) #sigmoid function used since its classification
    
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy']) # accuracy used due to classification
    model.fit(X, Y, epochs=10, batch_size=10)
    scores = model.evaluate(X, Y)
    return (model,scores)
 

def main():
    """
    Main function
    *Note Running NN net will take longer
    """
    path = "shot_pre.csv" # input file has been processed in R to convert characters to numeric 
    (dataset,header) = read_data(path) 

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, -1, -1)
    
    train_x = missing_values(train_x)
    test_x = missing_values(test_x)

    model_rf = random_forest_classifier(train_x,train_y)
    predictions_rf = model_rf.predict(test_x)
   
      
    print ("Train Accuracy :: ", accuracy_score(train_y, model_rf.predict(train_x)))
    
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions_rf))
    print (" Confusion matrix ", confusion_matrix(test_y, predictions_rf))
    
    train_x_scaled = preprocessing.scale(train_x) # need to scale all parameters for NNnet
    (model_nn,scores) = neural_network(train_x_scaled,train_y) #this step takes longer
    test_x_scaled = preprocessing.scale(test_x)
    predictions_nn = model_nn.predict(test_x_scaled)
    train_prediction_nn = [round(x[0]) for x in model_nn.predict(train_x_scaled)]
    test_predictions_nn = [round (x[0]) for x in predictions_nn]

    
    print ("Train Accuracy :: ", accuracy_score(train_y, train_prediction_nn))
    
    
    print ("Test Accuracy  :: ", accuracy_score(test_y, test_predictions_nn))
    print (" Confusion matrix ", confusion_matrix(test_y, test_predictions_nn))
    
    
    # to identify feature imporance
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [header[i] for i in indices])
    plt.savefig("bar.jpg")
    
    """
    Tring to visualizing the model using confusion matrix
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confusion_matrix(test_y, test_predictions_nn), fmt='', ax=ax,annot=True)
    plt.savefig("test.jpg")
    """
    
    
if __name__ == "__main__":
    main()
    
    