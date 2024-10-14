"""Code for solving problems a)b)c) for the terrain data in project 1."""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model

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

# Load the terrain
terrain1 = imread('SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Møsvatn Austfjell')
plt.imshow(terrain1, cmap='cividis')
plt.colorbar(label='Metres above sea level')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('terrain_unscaled.png', bbox_inches = 'tight')


y_min = 1500
y_max = 1600
x_min = 950
x_max = 1050

plt.figure()
plt.title('Møsvatn Austfjell')
plt.imshow(terrain1, cmap='cividis')
plt.colorbar(label='Metres above sea level')
plt.hlines(y_min, x_min, x_max, 'r')
plt.hlines(y_max, x_min, x_max, 'r')
plt.vlines(x_min, y_min, y_max, 'r')
plt.vlines(x_max, y_min, y_max, 'r')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('terrain_unscaled_box.png', bbox_inches = 'tight')



terrain_part = terrain1[y_min:y_max, x_min:x_max]
plt.figure()
plt.title('Section of terrain')
plt.imshow(terrain_part, cmap='cividis')
plt.colorbar(label='Metres above sea level')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('terrain_part_unscaled.png', bbox_inches = 'tight')

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
        RegLasso = linear_model.Lasso(lmbd,fit_intercept=True, max_iter=2000)
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

    plt.scatter(np.arange(1,len(beta)+1), beta, label=f'polynomial degree {p}')
    plt.errorbar(np.arange(1,len(beta)+1), beta, yerr=2*np.sqrt(var_beta), fmt="o", capsize=4)


plt.xlabel('i')
plt.ylabel('$\\beta_i$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('Beta_project1_OLS_terrain.png', bbox_inches = 'tight')

    
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
plt.savefig('MSE_p_project1_OLS_terrain.png', bbox_inches = 'tight')

#R2: 
plt.figure()
plt.title('OLS')
plt.plot(p_list, R2_train_list, 'o--', label='train')
plt.plot(p_list, R2_test_list, 'o--', label='test')
plt.xlabel('polynomial degree')
plt.ylabel('$R^2$')
plt.legend()
plt.savefig('R2_p_project1_OLS_terrain.png', bbox_inches = 'tight')

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
plt.savefig('MSE_p_project1_ridge_terrain.png', bbox_inches = 'tight')
    
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
plt.savefig('MSE_p_project1_lasso_terrain.png', bbox_inches = 'tight')


# Plot terrain from model
X = get_design_matrix(x,y,p)
z = X @ beta + y_scaler #add intercept
z = z*z_max #rescale back to original magnitude
zz = z.reshape(np.shape(xx))
 
# Plot the surface.
plt.figure()
plt.title('Approximation of terrain')
plt.imshow(zz, cmap='cividis')
plt.colorbar(label='Metres above sea level')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('OLS_terrain_approximated.png', bbox_inches='tight')
