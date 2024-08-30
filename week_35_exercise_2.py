"""
Week 35, Exercise 2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

noise_coeff = 0.1 #0.5, 1 

x = np.random.rand(100, 1)
y = 2.0+5*x*x+noise_coeff*np.random.randn(100,1)

#Write your own code (following the examples under the regression notes) 
#for computing the parametrization of the data set 
#fitting a second-order polynomial.

#create design matrix:
X = np.ones((len(x),3))
X[:,1] = x.flatten() #flatten() or x[:,0], otherwise dimensions are wrong
X[:,2] = x.flatten()**2

#Also made a function for generating a design matrix
def get_design_matrix(x, pol_deg):
    """Creates a design matrix for fitting polynomials of degree pol_deg"""
    X = np.ones((len(x), pol_deg+1))
    for p in range(0, pol_deg+1):
        X[:,p] = x.flatten()**p
    return X

p = 2
X = get_design_matrix(x, p)


XTX = np.linalg.inv(np.matmul(X.T,X)) #np.matmul or @ for matrix multiplication 
#find beta:
beta = np.matmul(np.matmul(XTX,X.T), y) #beta vector
#predict y:
y_pred = np.matmul(X,beta)

#sort for plotting line plots, otherwise they zigzag all over the place
i_sort = np.argsort(x.flatten()) #find indices that sort the x values
x_sort = x[i_sort] #sort x values
y_sort = y[i_sort] #sort y values so they match their x values
y_pred_sort = y_pred[i_sort]
plt.plot(x_sort.flatten(), y_sort.flatten(), 'ko',label='observed y')
plt.plot(x_sort.flatten(), y_pred_sort.flatten(), label='predicted y')

plt.xlabel('x')
plt.ylabel('y')


def get_MSE(y, y_pred):
    mean_SE = np.sum((y-y_pred)**2)/len(y)
    return mean_SE

def get_R2(y_true, y_pred):
    y_mean = np.mean(y_true)
    R2 = 1-(np.sum((y_true-y_pred)**2)/(np.sum((y_true-y_mean)**2)))
    return R2

MSE = get_MSE(y, y_pred)
R_sq = get_R2(y, y_pred)
print(f'MSE = {MSE}')
print(f'R^2 = {R_sq}')

#then sklearn
linreg = LinearRegression()
linreg.fit(X,y)
linreg_R_sq = linreg.score(X, y)
print('sklearn:')
print(f'MSE  = {mean_squared_error(y, y_pred)}')
print(f'R^2 = {linreg_R_sq}')

# Generate new x values to plot the sk-learn model
x_new = np.linspace(0, 1, len(x))
X_new = get_design_matrix(x_new, p)

y_pred_skl = linreg.predict(X_new)
plt.plot(x_new, y_pred_skl, "r--", label='sk-learn')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Second order polynomial fit')
plt.legend()
plt.savefig(f'exercise_2_{noise_coeff}.png')

















