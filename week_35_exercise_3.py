"""
Exercise 3, week 35. I used the code in the lecture notes as help.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Functions
def get_design_matrix(x, pol_deg):
    """Creates a design matrix for fitting polynomials of degree pol_deg"""
    X = np.ones((len(x), pol_deg+1))
    for p in range(0, pol_deg+1):
        X[:,p] = x.flatten()**p
    return X

def get_MSE(y, y_pred):
    mean_SE = np.sum((y-y_pred)**2)/len(y)
    return mean_SE

def get_R2(y_true, y_pred):
    y_mean = np.mean(y_true)
    R2 = 1-(np.sum((y_true-y_pred)**2)/(np.sum((y_true-y_mean)**2)))
    return R2

np.random.seed()
n = 100
# Make data set:
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
# Plot data:
plt.plot(x, y, 'ko')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Data for exercise 3')
plt.savefig('exercise_3_data.png')

# set up a design matrix defined by a fifth-order polynomial:
X = get_design_matrix(x, 5)

# split data set in training and test data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# matrix inversion to find beta:
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
# make prediction:
y_pred = X_train @ beta


MSE_train_list = []
MSE_test_list = []
p_list = []
#up to polynomial degree 15: 
for i in range(1,15+1):
    p = i #polynomial degree
    # set up a design matrix:
    X = get_design_matrix(x, p)
    # split data set in training and test data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # matrix inversion to find beta:
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    # make prediction:
    y_pred = X_train @ beta
    MSE_train = mean_squared_error(y_train, y_pred)#get_MSE(y_train, y_pred)
    #test model:
    y_pred = X_test @ beta 
    MSE_test = mean_squared_error(y_test, y_pred)#get_MSE(y_test, y_pred)
    #append lists for plotting
    MSE_train_list.append(MSE_train)
    MSE_test_list.append(MSE_test)
    p_list.append(p)

    plt.title('Polynomial fits of degree p')
    y_plot = X @ beta
    plt.plot(X[:,1], y_plot, '--',label=f'p = {p}')
    plt.legend()
    
# For which polynomial degree do you find an optimal MSE:
i_min = np.argmin(np.array(MSE_test_list))
p_min = p_list[i_min]
print(f'Smallest MSE in test data for polynomial degree {p_min}')


plt.savefig('exercise_3_polyfits.png')
#plot: 
plt.figure()
plt.plot(p_list, MSE_train_list, label='train')
plt.plot(p_list, MSE_test_list, label='test')
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig('MSE_p.png')
    
    
    
    
    
    
    
    







