"""
Code for solving problems f) for the Franke function data in project 1.
Implement the k-fold cross-validation algorithm (write your own code or
use the functionality of Scikit-Learn) and evaluate again the MSE function
resulting from the test folds.
Compare the MSE you get from your cross-validation code with the one you
got from your bootstrap code. Comment your results. Try 5 − 10 folds.
In addition to using the ordinary least squares method, you should include
both Ridge and Lasso regression.
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# for plots:
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#functions:
def get_MSE(y, y_pred):
    mean_SE = np.mean((y-y_pred)**2)
    return mean_SE

def get_R2(y_true, y_pred):
    y_mean = np.mean(y_true)
    R2 = 1-(np.sum((y_true-y_pred)**2)/(np.sum((y_true-y_mean)**2)))
    return R2

def get_design_matrix(x1,x2, pol_deg):
    """Creates a design matrix for fitting polynomials of degree pol_deg
    for two-dimensional data"""
    X = np.ones((len(x1), 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()
    poly = PolynomialFeatures(degree = pol_deg, include_bias=False)#no intercept
    X_poly = poly.fit_transform(X)
    return X_poly

def cross_validate(X, y, fitting_and_scoring_function, parameter, folds):
    """Takes X and y, a function which fits and data and scores the fit, 
    an additional tuning parameter and the number of k-folds. 
    Returns the mean of the MSE for all folds."""
    #parameter: lambda in the case of ridge and lasso, nothing in the case of OLS
    indexes = [i for i in range(np.shape(X)[0])]
    kf = KFold(n_splits=folds, shuffle=True)
    scores=[]
    error_list = []
    bias_list = []
    variance_list = []
    for train_indexes, test_indexes in kf.split(indexes):
        X_train = X[train_indexes,:]
        y_train = y[train_indexes]
        X_test = X[test_indexes,:]
        y_test = y[test_indexes]

        score, y_pred = fitting_and_scoring_function(X_train, X_test, y_train, y_test, parameter)
        scores.append(score)
        #aggregate all scores in cross validation to a single score
        error = np.mean((y_test - y_pred)**2) 
        bias = np.mean((y_test - np.mean(y_pred, keepdims=True))**2)
        variance = np.var(y_pred)
        error_list.append(error)
        bias_list.append(bias)
        variance_list.append(variance)    
    return np.mean(scores), np.mean(error_list), np.mean(bias), np.mean(variance)

def fit_OLS(X_train, X_test, y_train, y_test, l = 0):
    # l = 0 is only there to fit the same format as the ridge and LASSO functions
    OLS = LinearRegression(fit_intercept=True)
    OLS.fit(X_train, y_train)
    y_pred = OLS.predict(X_test)
    train_score = get_MSE(y_test, y_pred)  
    return train_score, y_pred

def fit_lasso(X_train, X_test, y_train, y_test, l):
    #Makes a lasso regression fit with training data, scores with test data
    # l: lambda
    lasso = Lasso(l)
    lasso.fit(X_train, y_train)
    #lasso_coeff = lasso.coef_ 
    y_pred = lasso.predict(X_test).reshape(-1,1)
    train_score = get_MSE(y_test, y_pred)
    #train_score = lasso.score(X_test,y_test) #R2
    return train_score, y_pred #.reshape(-1,1)

def fit_ridge(X_train, X_test, y_train, y_test, l):
    # l: lambda
    ridge = Ridge(l)
    ridge.fit(X_train, y_train)
    #lasso_coeff = lasso.coef_ 
    y_pred = ridge.predict(X_test)
    train_score = get_MSE(y_test, y_pred)
    #train_score = ridge.score(X_test,y_test) 
    return train_score, y_pred

# Make data set.
N = 1000
x = np.random.uniform(0,1, N)
y = np.random.uniform(0,1, N)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape) #added noise

maxdegree = 16
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

MSE_train_list = []
MSE_test_list = []
R2_train_list = []
R2_test_list = []
p_list = []

X = np.array([x,y]).T 
x = X
y = z.reshape(-1,1)

MSE_list_OLS = []
MSE_list_ridge = []
MSE_list_lasso = []

error_list_OLS = []
error_list_ridge = []
error_list_lasso = []

bias_list_OLS = []
bias_list_ridge = []
bias_list_lasso = []

variance_list_OLS = []
variance_list_ridge = []
variance_list_lasso = []

lambdas = np.logspace(-4, 1, 6)

MSE_ridge_array = np.zeros((len(np.arange(maxdegree-1)), len(lambdas)))
MSE_lasso_array = np.zeros((len(np.arange(maxdegree-1)), len(lambdas)))
deg_i = 0
MSE_list_OLS = []
for degree in range(1,maxdegree):
    lambda_i = 0
    MSE_deg_list = []
    MSE_list_ridge = []
    MSE_list_lasso = []
    
    for lmbd in lambdas:
        k = 10 #number of folds
        poly = PolynomialFeatures(degree = degree, include_bias=False)#no intercept
        X_poly = poly.fit_transform(X)

        # OLS: 
        MSE_OLS, error_OLS, bias_OLS, variance_OLS = cross_validate(X_poly, y, fit_OLS, parameter=0, folds=k)
        #MSE_list_OLS.append(MSE_OLS)
        error_list_OLS.append(error_OLS)
        bias_list_OLS.append(bias_OLS)
        variance_list_OLS.append(variance_OLS)
        # Ridge:
        MSE_ridge, error_ridge, bias_ridge, variance_ridge = cross_validate(X_poly, y, fit_ridge, parameter=lmbd, folds=k)
        MSE_list_ridge.append(MSE_ridge)
        error_list_ridge.append(error_ridge)
        bias_list_ridge.append(bias_ridge)
        variance_list_ridge.append(variance_ridge)
        # LASSO: 
        MSE_lasso, error_lasso, bias_lasso, variance_lasso = cross_validate(X_poly, y, fit_lasso, parameter=lmbd, folds=k)
        MSE_list_lasso.append(MSE_lasso)
        error_list_lasso.append(error_lasso)
        bias_list_lasso.append(bias_lasso)
        variance_list_lasso.append(variance_lasso)
        polydegree[degree] = degree

    MSE_list_OLS.append(MSE_OLS)
    MSE_ridge_array[deg_i,:] = np.asarray(MSE_list_ridge)
    MSE_lasso_array[deg_i,:] = np.asarray(MSE_list_lasso)
    deg_i += 1
        

plt.figure()
plt.plot(polydegree[1:], MSE_list_OLS, 'o--')
plt.title(f'OLS, n = {N}, number of k-folds = {k}')
plt.ylabel('MSE')
plt.xlabel('Polynomial degree')


plt.figure()
for i in range(np.shape(MSE_ridge_array)[1]):
    plt.plot(polydegree[1:], MSE_ridge_array[:,i], 'o--', label=f'$\\lambda$ = {lambdas[i]}')
plt.plot(polydegree[1:], MSE_list_OLS, 'ko--', label='OLS')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f'Ridge, n = {N}, number of k-folds = {k}')
plt.ylabel('MSE')
plt.xlabel('Polynomial degree')
plt.savefig('MSE_poldeg_lambda_ridge.png', bbox_inches = 'tight')

plt.figure()
for i in range(np.shape(MSE_lasso_array)[1]):
    plt.plot(polydegree[1:], MSE_lasso_array[:,i], 'o--', label=f'$\\lambda$ = {lambdas[i]}')
plt.plot(polydegree[1:], MSE_list_OLS, 'ko--', label='OLS')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f'LASSO, n = {N}, number of k-folds = {k}')
plt.ylabel('MSE')
plt.xlabel('Polynomial degree')
plt.savefig('MSE_poldeg_lambda_lasso.png', bbox_inches = 'tight')
