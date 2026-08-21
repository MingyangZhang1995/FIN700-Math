# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
from scipy import sparse

# generate random data with 1000 categories
# y = alpha_c + beta * x + epsilon, where alpha_c is the category fixed effect
N = 10000 # number of observations, each category has 10 observations
K = 1 # number of independent variables
P = 1000 # number of categories
np.random.seed(0)
# generate random independent variable x
x = np.random.randn(N, K)
# generate random category fixed effect alpha_c
alpha_c = np.random.randn(P, 1)
# generate random category assignment for each observation
category = np.random.randint(0, P, size=(N, 1))
# generate random error term epsilon
epsilon = np.random.randn(N, 1)
# generate dependent variable y
y = alpha_c[category] + x * 2 + epsilon
# create dummy variables for each category
z = np.eye(P)[category.flatten()] 

def estimate_ols(z, x, y):
   X = np.hstack((z, x))
   beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
   return beta_hat


def absorb_ols(z, x):
   z_sparse = sparse.csr_matrix(z)
   tilde_x = x - (z_sparse @ sparse.linalg.spsolve(z_sparse.T @ z_sparse, z_sparse.T @ x)).reshape(N,K)
   beta_hat = np.linalg.pinv(tilde_x.T @ tilde_x) @ tilde_x.T @ y
   return beta_hat

# count time for estimating OLS with dummy variables
import time
start_time = time.time()
beta_hat_ols = estimate_ols(z, x, y)
duration = time.time() - start_time
print(f"Time for estimating OLS with dummy variables: {duration:.2f} seconds")

# count time for estimating OLS with absorbing
start_time = time.time()
beta_hat_absorb = absorb_ols(z, x)
duration = time.time() - start_time
print(f"Time for estimating OLS with absorbing: {duration:.2f} seconds")
#
#
#
#
# plot y against tilde_x
import matplotlib.pyplot as plt
z_sparse = sparse.csr_matrix(z)
tilde_x = x - (z_sparse @ sparse.linalg.spsolve(z_sparse.T @ z_sparse, z_sparse.T @ x)).reshape(N,K)
plt.scatter(tilde_x, y, alpha=0.5)
plt.xlabel('tilde_x')
plt.ylabel('y')
plt.title('Scatter plot of y against tilde_x')
plt.show()
#
#
#
#
