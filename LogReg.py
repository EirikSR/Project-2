from functions import *
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import os

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import preprocessing
import scipy.integrate
import matplotlib.pyplot as plt
import scikitplot as skp
import matplotlib.cm as cm

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def costFunction(theta, X, y):
 
    m=len(y)

    predictions = sigmoid(np.dot(X, theta))
    
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
    cost = 1./m * sum(error)
    
    grad = 1./m * np.dot(X.transpose(), (predictions - y))
    
    return cost , grad

def gradientDescent(X,y,theta,alpha,):
	
    cost, grad = costFunction(theta,X,y)
    theta = theta - (alpha * grad)

    return theta

def predict(X, theta):
        input_ = np.dot(X, theta)
        output = sigmoid(input_)

        predict = lb.inverse_transform(output)

        return predict

def LogReg(X,y): 
    batch_size = 50
    itterations = 100000
    alpha = 0.001
    
    cost = -1
    theta =  np.random.uniform(-0.5,0.5,X.shape[1])
    theta_lst = []
    
    for i in xrange (itterations):

        if(batch_size < len(X)):
            indecies = np.random.choice(len(X), batch_size)
            X_batch = X_train[indecies, : ]
            y_batch = y_train[indecies]

        else:
            X_batch = X_train
            y_batch = y_train

        if (cost == -1):
            
            cost, theta = costFunction(theta, X_batch, y_batch)
        else:
            theta = gradientDescent(X_batch, y_batch, theta, alpha)

        if(i % 100 == 0):
            theta_lst.append(theta)
            
    return theta

X, y = setup_Design_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

y = y.reshape(-1, 1)

theta = LogReg(X_train, y_train)

lb = LabelBinarizer()   

y_test = lb.fit_transform(y_test)

pred = predict(X_test, theta)

R1, R2 = CumGain(pred, y_test)

print 'Default ratio: ', R1
print 'Not default ratio: ', R2

pred = predict(X_test, theta)

print 'Validation:'
print_confusion(y_test, pred)

pred = predict(X, theta)

print 'Full set:'
print_confusion(y, pred)





