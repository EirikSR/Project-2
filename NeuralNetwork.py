from functions import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

class NeuralNetwork:

	def __init__(self):
		#Initiating lists used in the neural network
		self.syn = []
		self.bias = []
		self.Error_lst = []
		self.Activ = []
		self.seed = 1
		self.lb = LabelBinarizer()

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def tanh(self, x):
		return 2/(1 + np.exp(-2*x)) - 1

	def linear(self, x):
		return x



	def feed_forward(self, X):
		
		#creating an instance of input layer for compyting

		Activ_temp = X


		for l in xrange(len(self.Activ)):
			
			Activ_temp = np.dot(Activ_temp, self.syn[l]) + self.bias[l]

			if (self.func_lst[l] == 'sigmoid'):
				func = self.sigmoid

			elif (self.func_lst[l] == 'tanh'):
				func = self.tanh

			elif (self.func_lst[l] == 'linear'):
				func = self.linear

			Activ_temp = func(Activ_temp)

			self.Activ[l] = Activ_temp

		return self

	def backpropagation(self, X, y):
		
		#Error_Y = np.expand_dims(y, axis=1)
		

		if (self.func_lst[-1] == 'linear'):
				error_output = (self.Activ[-1] - y)/X.shape[0]
				#print sum(error_output)
		else:
			error_output = self.Activ[-1] - y

		self.Error_lst[-1] = error_output
		error = error_output
		
		#Looping through the hidden layers backwards to calculate error
		for l in range(len(self.Activ)-2, -1, -1):

			#Error calculated according to activation function used
			if (self.func_lst[l] == 'sigmoid'):
				error = np.dot(error, np.transpose(self.syn[l+1])) * self.Activ[l]*(1- self.Activ[l])

			elif (self.func_lst[l] == 'tanh'):

				error = np.dot(error, np.transpose(self.syn[l+1])) * (1 - (self.Activ[l])**2)
			self.Error_lst[l] = error

		
		#Transposing first in order to solve crashing
		a =  np.transpose(self.Activ[-2])

		self.syn[-1] -= self.eta * (np.dot(a, self.Error_lst[-1]) + self.lmda*self.syn[-1])
		self.bias[-1] -= self.eta * np.sum(self.Error_lst[-1],  axis=0)

		for l in range(len(self.Activ)-2, 0, -1):
			
			activ_transposed = np.transpose(self.Activ[l-1])
			
			self.syn[l] -= self.eta * (np.dot(activ_transposed, self.Error_lst[l]) + self.lmda*self.syn[l])
			self.bias[l] -= self.eta * np.sum(self.Error_lst[l], axis=0)

		self.syn[0] -= self.eta * (np.dot(np.transpose(X), self.Error_lst[0]) + self.lmda*self.syn[0])
		self.bias[0] -= self.eta * np.sum(self.Error_lst[0],  axis=0)

		




	def run_model(self 
					,X
					,y
					,neurons
					,func_lst
					,itterations = 10000
					,batch_size = 20
					,eta = 0.1
					,lmda = 0.001
					,test_size = 0.5
					,credit = True):
		
		self.inputs = X.shape[0]
		self.features = X.shape[1]
		self.lmda = lmda
		length = len(neurons)

		self.eta = eta
		self.func_lst = func_lst
		
		self.syn.append(np.random.random((self.features, neurons[0])) - 0.5)

		for l in xrange(length-1):
			self.syn.append(np.random.random((neurons[l], neurons[l+1])) - 0.5)

		for l in xrange(length):
			self.bias.append(np.random.random((neurons[l])) -0.5)

		for i in xrange(length):
			self.Activ.append(0)
			self.Error_lst.append(0)
		if (credit == True):
			y = self.lb.fit_transform(y)
		
		
		for epoch in range(itterations +1):
			if (epoch % 1000 == 0):
				print epoch

			if (batch_size < X.shape[0]):

				indecies = np.random.choice(X.shape[0], batch_size)
				X_batch = X[indecies, :]
				y_batch = y[indecies]

			else:
				X_batch = X
				y_batch = y

			self.feed_forward(X_batch)
			self.backpropagation(X_batch, y_batch)

			
			#predict = self.predict(X_test)
			#y_val = lb.inverse_transform(y_test)
			score =1 # accuracy_score(y_val, predict)
			if (epoch % 1000 == 0 and credit):
				
				predict = self.predict(X, y, credit)
				acc = accuracy_score(predict, y)
				#print(confusion_matrix(y_train, predict))
		return self


	def predict(self, X, y, credit):
		self.feed_forward(X)
		output = self.Activ[-1]
		
		
		if (credit):
			predict = self.lb.inverse_transform(output)
			"""print sum(predict), sum(y)
									
			good = 0
			bad = 0
			for x, y in zip(predict, y):
				
				if(x != 0):
					if(x == y):
						good += 1
					else:
						bad += 1

			print good, bad, good + bad"""

		else:
			predict = output
		return predict


X, y = setup_Design_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

neurons = [25, 25, 2]
func_lst = ['tanh', 'tanh','linear']

nn = NeuralNetwork()

nn.run_model(X_train, y_train, neurons, func_lst, credit = True)

predict = nn.predict(X_test, y_test, True)

R1, R2 = CumGain(predict, y_test)

print 'Default ratio: ', R1
print 'Not default ratio: ', R2

print "Validation:"
print_confusion(y_test, predict)


predict = nn.predict(X, y, True)

print "Full set:"
print_confusion(y, predict)
