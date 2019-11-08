import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
def nonlin(x, deriv = False):
	if(deriv == True):
		
		return (x*(1-x))

	else:
		return 1/(1 + np.exp(-x))

np.random.seed(0)


nanDict = {}

df = pd.read_excel("data.xls", header=0, skiprows=0, index_col=0, na_values=nanDict)

X = df.loc[0:1000, df.columns != 'default payment next month'].values
y = df.loc[0:1000, df.columns == 'default payment next month'].values



onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough").fit_transform(X)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = x_scaled

print X.shape, y.shape


syn0 = 2*np.random.random((31, 1000)) 
syn1 = 2*np.random.random((1000, 1)) 

for j in xrange(100):

	l0 = X
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))

	l2_error = y - l2
	
	if ((j % 10) == 0):
		print "error: " + str(np.mean(np.abs(l2_error)))

	l2_delta = l2_error*nonlin(l2, deriv=True)
	
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print l2 -y