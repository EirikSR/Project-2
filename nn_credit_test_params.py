from a import *
from functions import *

X, y = setup_Design_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

neurons = [25, 25, 2]
func_lst = ['tanh', 'tanh','linear']

nn = NeuralNetwork()


lamda = np.logspace(0,-8,20)
eta = np.logspace(0,-6,20)

lst = []
lst_val = []

temp = [0]

for i in lamda:
	temp.append(i)
lst.append(temp)
lst_val.append(temp)
for a in eta:
	temp = []
	temp.append(a)
	temp_val = []
	temp_val.append(a)


	for b in lamda:

		nn = NeuralNetwork()
		nn.run_model(X_train, y_train, neurons, func_lst, eta = a, lmda = b, credit = True)

		predict = nn.predict(X_test, y_test, True)
		acc2 = accuracy_score(predict, y_test)

		predict = nn.predict(X_train, y_train, True)
		acc1 = accuracy_score(predict, y_train)

		#lst.append([a, b, acc1, acc2])
		print (a, b, acc1, acc2)
		temp.append(acc1)
		temp_val.append(acc2)
	lst.append(temp)
	lst_val.append(temp_val)
import matplotlib.pyplot as plt
print lst

arr = np.asarray(lst)
arr2 = np.asarray(lst_val)
np.savetxt("foo1.csv", arr, delimiter=",")
np.savetxt("foo2.csv", arr2, delimiter=",")