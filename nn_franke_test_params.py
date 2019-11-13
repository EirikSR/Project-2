from NeuralNetwork import *
from functions import *
import math


#X, y = setup_Design_matrix('s')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)


x = np.arange(0, 1, 0.001)
y = np.arange(0, 1, 0.001)
x, y = np.meshgrid(x,y)
n = np.size(x, 0)

noisy_x = x + 0.05 * np.random.randn(len(x))
noisy_y = y + 0.05 * np.random.randn(len(y))

z = Franke(x, y, False)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)

noisy_x = noisy_x.reshape(-1, 1)
noisy_y = noisy_y.reshape(-1, 1)

dataset = np.concatenate((noisy_x, noisy_y), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(dataset, z, test_size=0.25)


neurons = [25, 25, 10, 1]
func_lst = ['tanh', 'tanh', 'tanh', 'linear']

lamda = np.logspace(0,-8, 20)
eta = np.logspace(0,-6, 20)


lst = []
temp = [0]

for i in lamda:
	temp.append(i)
lst.append(temp)

for a in eta:
	temp = []
	temp.append(a)

	for b in lamda:

		nn = NeuralNetwork()
		nn.run_model(X_train, y_train, neurons, func_lst, eta = a, lmda = b, credit = False, itterations = 2500)

		pred = nn.predict(X_test, y_test, False)

		acc =  R2(pred, y_test)
		acc = float(acc)
		if(acc < -1 or acc > 1 or math.isnan(acc)):
			acc = float('NaN')

		#lst.append([a, b, acc1, acc2])
		print (a, b, float(acc))
		temp.append(acc)
	lst.append(temp)
	
print lst


arr = np.asarray(lst)
print arr

np.savetxt("A2.csv", arr, delimiter=",")
