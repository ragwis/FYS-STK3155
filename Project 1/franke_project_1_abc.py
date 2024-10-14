"""
Code for solving problems a)b)c) for the Franke function data in project 1.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
    X[:,0] = x.flatten()
    X[:,1] = y.flatten()
    poly = PolynomialFeatures(degree = pol_deg, include_bias=False)#no intercept
    X_poly = poly.fit_transform(X)
    return X_poly

def get_beta_ridge(X, y, lambd):
    p = np.shape(X)[1] 
    I = np.eye(p,p)
    beta = np.linalg.pinv(X.T @ X+lambd*I) @ X.T @ y
    return beta


# Make data:
N = 1000
x = np.random.uniform(0,1, N)
y = np.random.uniform(0,1, N)

xx, yy = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y) + 0.1*np.random.normal(0, 1, x.shape) #added noise
zz = FrankeFunction(xx, yy)

MSE_train_list = []
MSE_test_list = []
R2_train_list = []
R2_test_list = []
p_list = []

MSE_train_list_ridge = []
MSE_test_list_ridge = []
R2_train_list_ridge = []
R2_test_list_ridge = []

MSE_train_list_lasso = []
MSE_test_list_lasso = []
R2_train_list_lasso = []
R2_test_list_lasso = []

plt.figure()
plt.title('OLS')

#up to polynomial degree 15: 
for i in range(1,5+1):
    p = i #polynomial degree
    # set up a design matrix:
    X = get_design_matrix(x,y,p)   
    # split data set in training and test data:
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
    
    X_train_mean = np.mean(X_train,axis=0)
    #Center by removing mean from each feature
    X_train_scaled = X_train - X_train_mean 
    X_test_scaled = X_test - X_train_mean
    #The model intercept (called y_scaler) is given by the mean of the target variable (IF X is centered)
    #Remove the intercept from the training data.
    y_scaler = np.mean(y_train)           
    y_train_scaled = y_train - y_scaler 
    # matrix inversion to find beta:
    beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train_scaled
    var_beta = np.diagonal(np.var(y_train_scaled)*np.linalg.inv(X_train_scaled.T @ X_train_scaled))
    intercept = y_scaler - X_train_mean@beta
    # make prediction:
    y_pred = X_train_scaled @ beta + y_scaler
    MSE_train = get_MSE(y_train, y_pred)
    R2_train = get_R2(y_train, y_pred)
    #test model:
    y_pred = X_test_scaled @ beta  + y_scaler
    MSE_test = get_MSE(y_test, y_pred)
    R2_test = get_R2(y_test, y_pred)
    #append lists for plotting
    MSE_train_list.append(MSE_train)
    MSE_test_list.append(MSE_test)
    R2_train_list.append(R2_train)
    R2_test_list.append(R2_test)
    p_list.append(p)
    #----------------Ridge---------------------
    lambdas = np.logspace(-4, 1, 6)
    temp_test_MSE_list = []
    temp_train_MSE_list = []
    temp_test_R2_list = []
    temp_train_R2_list = []
    
    temp_test_MSE_list_lasso = []
    temp_train_MSE_list_lasso = []
    temp_test_R2_list_lasso = []
    temp_train_R2_list_lasso = []
    for j in range(len(lambdas)):
        lmbd = lambdas[j]
        print(f'Lambda = {lmbd}')
        beta_ridge = get_beta_ridge(X_train_scaled, y_train_scaled, lmbd)
        #intercept_ = y_scaler - X_train_mean@beta_ridge #The intercept can be shifted so the model can predict on unc
        # Add intercept to prediction
        y_pred_test_ridge = X_test_scaled @ beta_ridge + y_scaler
        MSE_test_ridge = get_MSE(y_test, y_pred_test_ridge)
        R2_test_ridge = get_R2(y_test, y_pred_test_ridge)
        
        temp_test_MSE_list.append(MSE_test_ridge)
        temp_test_R2_list.append(R2_test_ridge)
        y_pred_train_ridge = X_train_scaled @ beta_ridge + y_scaler
        MSE_train_ridge = get_MSE(y_train, y_pred_train_ridge)
        R2_train_ridge = get_R2(y_train, y_pred_train_ridge)
        
        temp_train_MSE_list.append(MSE_train_ridge)
        temp_train_R2_list.append(R2_train_ridge)
        #----------lasso-----------
        RegLasso = linear_model.Lasso(lmbd,fit_intercept=True)
        RegLasso.fit(X_train_scaled,y_train_scaled)
        # and then make the prediction
        y_pred_test_lasso = RegLasso.predict(X_test_scaled) + y_scaler
        y_pred_train_lasso = RegLasso.predict(X_train_scaled) + y_scaler
        MSE_test_lasso = get_MSE(y_test, y_pred_test_lasso)
        R2_test_lasso = get_R2(y_test, y_pred_test_lasso)
        MSE_train_lasso = get_MSE(y_train, y_pred_train_lasso)
        R2_train_lasso = get_R2(y_train, y_pred_train_lasso)
        
        temp_train_MSE_list_lasso.append(MSE_train_lasso)
        temp_train_R2_list_lasso.append(R2_train_lasso)
        temp_test_MSE_list_lasso.append(MSE_test_lasso)
        temp_test_R2_list_lasso.append(R2_test_lasso)
        
    MSE_test_list_ridge.append(temp_test_MSE_list)
    MSE_train_list_ridge.append(temp_train_MSE_list)
    R2_test_list_ridge.append(temp_test_R2_list)
    R2_train_list_ridge.append(temp_train_R2_list)
    
    MSE_test_list_lasso.append(temp_test_MSE_list_lasso)
    MSE_train_list_lasso.append(temp_train_MSE_list_lasso)
    R2_test_list_lasso.append(temp_test_R2_list_lasso)
    R2_train_list_lasso.append(temp_train_R2_list_lasso)

    #Beta:    
    plt.scatter(np.arange(1,len(beta)+1), beta, label=f'polynomial degree {p}')
    plt.errorbar(np.arange(1,len(beta)+1), beta, yerr=2*np.sqrt(var_beta), fmt="o", capsize=4)
    #plt.plot(p_list, R2_test_list, label='test')

plt.xlabel('i')
plt.ylabel('$\\beta_i$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('Beta_project1_OLS.png', bbox_inches = 'tight')

    
# OLS
# For which polynomial degree do you find an optimal MSE:
i_min = np.argmin(np.array(MSE_test_list))
p_min = p_list[i_min]
print(f'Smallest MSE in test data for polynomial degree {p_min}')

#plot: 
plt.figure()
plt.title('OLS')
plt.plot(p_list, MSE_train_list, 'o--', label='train')
plt.plot(p_list, MSE_test_list, 'o--', label='test')
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig('MSE_p_project1_OLS.png', bbox_inches = 'tight')

#R2: 
plt.figure()
plt.title('OLS')
plt.plot(p_list, R2_train_list, 'o--', label='train')
plt.plot(p_list, R2_test_list, 'o--', label='test')
plt.xlabel('polynomial degree')
plt.ylabel('$R^2$')
plt.legend()
plt.savefig('R2_p_project1_OLS.png', bbox_inches = 'tight')

#----------------------------------------------------------
#     Ridge
#----------------------------------------------------------
plt.figure()
for i in range(len(lambdas)):
    lam = lambdas[i]
    plt.title('Ridge')
    #plt.plot(p_list, np.asarray(MSE_train_list_ridge)[:,i], 'o--', label=f'train, $\\lambda$ = {lam}')
    plt.plot(p_list, np.asarray(MSE_test_list_ridge)[:,i], 'o--', label=f'test, $\\lambda$ = {lam}')
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('MSE_p_project1_ridge.png', bbox_inches = 'tight')
    
#----------------------------------------------------------
#     LASSO
#----------------------------------------------------------
plt.figure()
for i in range(len(lambdas)):
    lam = lambdas[i]
    plt.title('LASSO')
    #plt.plot(p_list, MSE_train_list_lasso[i], 'o--', label=f'train, $\\lambda$ = {lam}')
    plt.plot(p_list, np.asarray(MSE_test_list_lasso)[:,i], 'o--', label=f'test, $\\lambda$ = {lam}')
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('MSE_p_project1_lasso.png', bbox_inches = 'tight')


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
xx, yy = np.meshgrid(x,y) 
x = np.ravel(xx)
y = np.ravel(yy)
X = get_design_matrix(x,y,p)
z = X @ beta + y_scaler
zz = z.reshape(np.shape(xx))
 
# Plot the surface:
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, pad = 0.1)
plt.savefig('OLS_franke_approximated.png', bbox_inches='tight')