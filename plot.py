from functions import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from decimal import Decimal

data = np.genfromtxt('A2.csv', delimiter=',') #Reads csv generated by nn_credit_test_params.py and nn_frange_test_params.py

print data

X = data[0]
x = []
y = []
print X
data = np.delete(data, 0, 0)

for i in X:
	print i
	x.append("{:.2e}".format(float(i)))

for i in data:
	y.append("{:.2e}".format(i[0]))

data = data[:,1:]


ax = sns.heatmap(data, linewidth=0.5)

plt.title("R2 as heatmap")

plt.yticks(range(len(y)), y, size='small', rotation='horizontal')
plt.ylabel('Learning Rate',rotation= 'vertical')

plt.xticks(range(len(x)), x, size='small', rotation='vertical')
plt.xlabel('Reactivation', rotation=0)

plt.show()