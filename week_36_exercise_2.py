"""
Week 36, exercise 2, with code from exercise 3, week 35 as a basis.

Compute the mean squared error for ordinary least squares and 
Ridge regression first for a polynomial of degree five with data points 
n = 100 and five selected values of lambda = [0.0001, 0.001, 0.01, 0.1, 1.0]. 
Compute thereafter the mean squared error for the same values of lambda
for polynomials of degree ten and 15. 
Discuss your results for the training MSE and test MSE with 
Ridge regression and ordinary least squares.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for plots:
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Functions
def get_design_matrix(x, pol_deg):
    """Creates a design matrix for fitting polynomials of degree pol_deg"""
    X = np.ones((len(x), pol_deg+1))
    for p in range(0, pol_deg+1):
        X[:,p] = x.flatten()**p
    return X

def get_design_matrix_no_intercept(x, pol_deg):
    """Creates a design matrix without an intercept column for 
    fitting polynomials of degree pol_deg"""
    X = np.zeros((len(x), pol_deg-1)) # no intercept column
    for p in range(1, pol_deg):
        X[:,p-1] = x.flatten()**p
        
    return X

def get_beta_OLS(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def get_beta_ridge(X, y, lambd):
    p = np.shape(X)[1] 
    I = I = np.eye(p,p)
    beta = np.linalg.pinv(X.T @ X+lambd*I) @ X.T @ y
    return beta

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
plt.title(r'Data for exercise 2')
plt.savefig('week_36_exercise_2_data.png')


MSE_train_list_OLS = []
MSE_test_list_OLS = []

MSE_train_list_ridge = []
MSE_test_list_ridge = []

p_list = []
#up to polynomial degree 15: 
poly_range = range(5,15+1, 5)
for i in range(len(poly_range)): #5, 10, 15 #CREATE FUNCTION INSTEAD!!! And call function in loop!
    p = poly_range[i] #polynomial degree
    # set up a design matrix without an intercept column:
    X = get_design_matrix_no_intercept(x, p)
    # split data set in training and test data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # center design matrix: 
    X_train_mean = np.mean(X_train,axis=0)
    #Center by removing mean from each feature
    X_train_scaled = X_train - X_train_mean 
    X_test_scaled = X_test - X_train_mean
    #The model intercept (called y_scaler) is given by the mean of the 
    # target variable (IF X is centered).
    #Remove the intercept from the training data:
    y_scaler = np.mean(y_train)           
    y_train_scaled = y_train - y_scaler 
    
    y_test_scaled = y_test - y_scaler #??
    #-----------OLS---------------------
    # matrix inversion to find OLS beta:
    beta_OLS = get_beta_OLS(X_train_scaled, y_train_scaled)
    # make prediction and add intercept:
    y_pred_OLS = X_train_scaled @ beta_OLS + y_scaler
    MSE_train_OLS = mean_squared_error(y_train_scaled, y_pred_OLS)#get_MSE(y_train, y_pred)
    #test model and add intercept:
    y_pred_test_OLS = X_test_scaled @ beta_OLS + y_scaler
    MSE_test_OLS = mean_squared_error(y_test_scaled, y_pred_test_OLS)#get_MSE(y_test, y_pred)
    #append lists for plotting
    MSE_train_list_OLS.append(MSE_train_OLS)
    MSE_test_list_OLS.append(MSE_test_OLS)
    p_list.append(p)
    
    print(f'Polynomial degree: {p}')
    print('OLS:')
    print(f'MSE test: {MSE_test_OLS}')
    print(f'MSE train: {MSE_train_OLS}')
    

    plt.title('Polynomial fits of degree p')
    intercept = np.mean(y_scaler - X_train_mean @ beta_OLS)
    y_plot = X @ beta_OLS + intercept#y_scaler
    plt.plot(X[:,0], y_plot, '--',label=f'p = {p}')
    plt.legend()
    #----------------Ridge---------------------
    # Decide which values of lambda to use
    lambdas = np.logspace(-4, 0, 5)
    temp_test_MSE_list = []
    temp_train_MSE_list = []
    print('-----------------------------------')
    print('Ridge:')
    for j in range(len(lambdas)):
        lmbd = lambdas[j]
        print(f'Lambda = {lmbd}')
        beta_ridge = get_beta_ridge(X_train_scaled, y_train_scaled, lmbd)
        intercept_ = y_scaler - X_train_mean@beta_ridge #The intercept can be shifted so the model can predict on unc
        # Add intercept to prediction
        y_pred_test_ridge = X_test_scaled @ beta_ridge + y_scaler
        MSE_test_ridge = get_MSE(y_test_scaled, y_pred_test_ridge)
        #fix this, iverwritten for each p
        temp_test_MSE_list.append(MSE_test_ridge)
        
        y_pred_train_ridge = X_train_scaled @ beta_ridge + y_scaler
        MSE_train_ridge = get_MSE(y_train_scaled, y_pred_train_ridge)
        #fix this, iverwritten for each p
        temp_train_MSE_list.append(MSE_train_ridge)
        # For which lambda do you find an optimal MSE: 
        #print(f'Lambda: {lmbd}')
        #print(f'beta: {beta_ridge}')
        #print(f'MSE: {MSE_test_ridge}')
        print(f'MSE test: {MSE_test_ridge}')
        print(f'MSE train: {MSE_train_ridge}')
    MSE_test_list_ridge.append(temp_test_MSE_list)
    MSE_train_list_ridge.append(temp_train_MSE_list)
    print('-----------------------------------')
# For which polynomial degree do you find an optimal MSE:
i_min = np.argmin(np.array(MSE_test_list_OLS))
p_min = p_list[i_min]
print(f'Smallest MSE in test data for polynomial degree {p_min}')




plt.savefig('week_36_2_polyfits.png')
#plot: 
plt.figure()
plt.plot(p_list, MSE_train_list_OLS, label='train')
plt.plot(p_list, MSE_test_list_OLS, label='test')
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig('week_36_MSE_p.png')