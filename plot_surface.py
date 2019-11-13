from NeuralNetwork import *
from functions import *

x = np.arange(0, 1, 0.002)
y = np.arange(0, 1, 0.002)
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



noisy = True

if noisy:
	dataset = np.concatenate((noisy_x, noisy_y), axis = 1)

else:
	dataset = np.concatenate((x, y), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(dataset, z, test_size=0.25)


neurons = [25, 25, 10,  1]
func_lst = ['tanh', 'tanh', 'tanh', 'linear']

nn = NeuralNetwork()
nn.run_model(X_train, y_train, neurons, func_lst, eta = 0.01, lmda = 0.0001, credit = False)

predict = nn.predict(X_test, y_test, False)


print "R2 test: ", R2(y_test, predict)

predict = nn.predict(dataset, z, False)

print "R2 train: ", R2(z, predict)

x = np.arange(0, 1, 0.002)
y = np.arange(0, 1, 0.002)
x, y = np.meshgrid(x,y)

predict = predict.reshape(len(x), len(x))



z = z.reshape(len(x), len(x))
print x.shape, y.shape, z.shape

plotSurface(x, y, z)
plotSurface(x, y, predict)
plotSurface(x, y, predict-z)