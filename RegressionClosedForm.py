


# @copy by sobhan siamak
import numpy as np
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

#Calculate Error
def MSE(theta0, theta1, Dataset):
    tError = 0
    for i in range(0, len(Dataset)):
        x = Dataset[i, 0]
        y = Dataset[i, 1]
        tError += (y - (theta1 * x + theta0)) ** 2
    return tError /(float(len(Dataset)))

# read train data
# train = pd.read_csv('train.csv', delimiter=',', header=None)
train = genfromtxt('train.csv', delimiter=',', skip_header=True)
test = genfromtxt('test.csv', delimiter=',', skip_header=True)
x = array(train[:,0])
x1 = x.reshape(len(x), 1)
y = array(train[:,1])
xtest = array(test[:,0])
ytest = array(test[:,1])


plt.scatter(x1,y, label='Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Closed Form Regression Line:')
# plt.show()

x = np.insert(x1, 0, 1, axis=1)  # add one columns of 1 as bias
theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print('in closed form solution:')
print('theta0 is :', theta[0])
print('theta1 is :', theta[1])
print("the vector of theta is:")
print(theta)

yhat = x.dot(theta)
plt.plot(x1, yhat,label='Regression-Line', color='r')
plt.legend()
plt.show()
# print(yhat.reshape(len(yhat),1))
# print(yhat)
theta0 = theta[0]
theta1 = theta[1]
print("MSE error from Train Data is :",MSE(theta0, theta1, train))
print("MSE error from Test Data is :", MSE(theta0, theta1, test))




