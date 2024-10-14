"""Code for solving problems e) for the terrain data in project 1."""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.utils import resample

# for plots:
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

#functions:
def get_MSE(y, y_pred):
    mean_SE = np.sum((y-y_pred)**2)/len(y)
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

np.random.seed(2018)
n_bootstraps = 100
maxdegree = 15

MSE_train_list = []
MSE_test_list = []
R2_train_list = []
R2_test_list = []
p_list = []

y_min = 1500
y_max = 1600
x_min = 950
x_max = 1050

# Load the terrain
terrain1 = imread('SRTM_data_Norway_2.tif')
terrain_part = terrain1[y_min:y_max, x_min:x_max]
y = np.arange(np.shape(terrain_part)[0])
x = np.arange(np.shape(terrain_part)[1])
y = y/np.max(y)
x = x/np.max(x)
yy, xx = np.meshgrid(y, x)

y = yy.ravel()
x = xx.ravel()
z = terrain_part.ravel()
z_max = np.max(z)
z = z/z_max

X = np.array([x,y]).T 
x = X
y = z
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
for degree in range(1,maxdegree):
    poly = PolynomialFeatures(degree = degree, include_bias=False)#no intercept
    X_train = poly.fit_transform(x_train)
    X_test = poly.fit_transform(x_test)
    model = LinearRegression(fit_intercept=True)   
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(X_train, y_train)
        y_ = y_.reshape(-1,1)
        model.fit(x_, y_)
        y_pred[:, i] = model.predict(X_test).ravel()

    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

plt.figure()
plt.plot(polydegree[1:], error[1:], 'ro-', label='Error')
plt.plot(polydegree[1:], bias[1:], 'o--',label='Bias')
plt.plot(polydegree[1:], variance[1:], 'o--',label='Variance')
plt.xlabel('Polynomial degree')
plt.title(f'OLS, n = {len(z)}, number of bootstraps = {n_bootstraps}')
plt.legend()
plt.savefig('OLS_bootstrap_terrain_bias_var.png')


