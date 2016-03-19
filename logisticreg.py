import numpy as np
import getdata as gd
import csv as csv
import matplotlib.pyplot as plt

# Hypothesis; returns np.array
def hyp(X,theta):
	return 1/(1 + np.exp(-np.matmul(X,theta).A1))

# Cost function J; returns scalar
def J(X,theta,y):
	return (-1/len(X))*(np.dot(y,np.log(hyp(X,theta))) + np.dot(1-y,np.log(1-hyp(X,theta))))

# Gradient of cost function; returns np.array
def G(X,theta,y):
	return (1/len(X))*(np.matmul(X.T,np.matrix(hyp(X,theta) - y).T))


# Training model
print('Loading training data...')
train_file = 'train.csv'
train_data = gd.get_features_array(train_file)

m = len(train_data)
X = np.matrix(np.c_[np.ones(m),train_data])		# features MATRIX
theta = np.matrix(np.zeros(X[0].size)).T 		# initial theta MATRIX(column vec)
y = gd.get_survived(train_file) 				# solutions ARRAY

threshold = 10e-15
old_cost = 0
cost = 100
maxits = int(10e6)								# just in case we don't converge
its = 0
cost_array = []
alpha = 0.0035									# learning rate

print('Training model...')
while(abs(cost-old_cost) >= threshold and its < maxits):
	its+= 1
	old_cost = cost
	cost = J(X,theta,y)
	cost_array.append([its,cost])
	theta-= (alpha*G(X,theta,y))


print('Loading test file...')
test_file = 'test.csv'
test_data = gd.get_features_array(test_file)
ids = gd.get_ids(test_file)

m = len(test_data)
X = np.matrix(np.c_[np.ones(m),test_data])

print('Guessing test solutions...')
test_sol = np.floor(hyp(X,theta) + 0.5).astype(int)

output = np.c_[ids,test_sol]

prediction_file = 'logreg_predictions.csv'
p = csv.writer(open(prediction_file,'w'))
p.writerow(['PassengerId','Survived'])
for row in output:
	p.writerow(row)

print('Done')

print('Last cost change: ', old_cost-cost)
print('Iterations: ', its)

cost_array = np.array(cost_array)
plt.plot(cost_array[:,0],cost_array[:,1])
plt.savefig('costplot.png')



