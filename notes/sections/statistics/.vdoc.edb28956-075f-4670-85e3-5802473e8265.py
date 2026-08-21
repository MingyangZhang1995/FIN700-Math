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
#
#
import numpy as np
from scipy import sparse
import timeit

# generate random data with 3000 categories
# y = alpha_c + beta * x + epsilon, where alpha_c is the category fixed effect
N = 30000 # number of observations, each category has 10 observations
K = 1 # number of independent variables
P = 3000 # number of categories
np.random.seed(0)
# generate random independent variable x
x = np.random.randn(N, K)
# generate category fixed effect alpha_c
alpha_c = np.arange(P) * 0.01
# generate category assignment for each observation
# no left out category, each category has N//P observations
category = np.repeat(np.arange(P), N//P).reshape(N, 1)
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
t_ols = timeit.repeat('estimate_ols(z, x, y)', globals=globals(), number=1, repeat=5)
t_ols = np.array(t_ols)
mean_time_ols = np.mean(t_ols)
std_time_ols = np.std(t_ols)
reps_ols = len(t_ols)
print(f"Time for estimating OLS with dummy variables: {mean_time_ols:.2f} seconds (± {std_time_ols:.2f}) from {reps_ols} repetitions")

# count time for estimating OLS with absorbing
t_absorb = timeit.repeat('absorb_ols(z, x)', globals=globals(), number=1, repeat=10)
t_absorb = np.array(t_absorb)
mean_time_absorb = np.mean(t_absorb)
std_time_absorb = np.std(t_absorb)
reps_absorb = len(t_absorb)
print(f"Time for estimating OLS with absorbing: {mean_time_absorb:.2f} seconds (± {std_time_absorb:.2f}) from {reps_absorb} repetitions")
#
#
#
#
#
import matplotlib.pyplot as plt
z_sparse = sparse.csr_matrix(z)
tilde_x = x - (z_sparse @ sparse.linalg.spsolve(z_sparse.T @ z_sparse, z_sparse.T @ x)).reshape(N,K)
tilde_y = y - (z_sparse @ sparse.linalg.spsolve(z_sparse.T @ z_sparse, z_sparse.T @ y)).reshape(N,1)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(x, y, alpha=0.5)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Scatter plot of y against x')
# Scatter plot of tilde_y against tilde_x
axs[1].scatter(tilde_x, tilde_y, alpha=0.5)
axs[1].set_xlabel(r'$\tilde{x}$')
axs[1].set_ylabel(r'$\tilde{y}$')
axs[1].set_title(r'Scatter plot of $\tilde{y}$ against $\tilde{x}$')
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| fig-align: center
import matplotlib.pyplot as plt

def draw_neural_network(layer_sizes):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Compute spacing parameters
    v_spacing = 1.0 / max(layer_sizes)
    h_spacing = 1.0 / (len(layer_sizes) + 1)

    # Draw nodes
    node_positions = []
    for i, layer_size in enumerate(layer_sizes):
        layer_nodes = []
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for j in range(layer_size):
            # Calculate coordinates for each node
            x = (i + 1) * h_spacing
            y = 0.5 + layer_top - j * v_spacing
            layer_nodes.append((x, y))
            
            # Draw the node circles
            circle = plt.Circle((x, y), v_spacing / 4.0, color='skyblue', ec='black', zorder=4)
            ax.add_artist(circle)
            
            # Label layers roughly
            if j == 0:
                layer_type = "Input" if i == 0 else ("Output" if i == len(layer_sizes)-1 else "Hidden")
                ax.text(x, 0.9, f"{layer_type}\nLayer", ha='center', va='bottom', fontsize=12, fontweight='bold')
        node_positions.append(layer_nodes)

    # Draw edges (connections)
    for i in range(len(layer_sizes) - 1):
        for node1 in node_positions[i]:
            for node2 in node_positions[i+1]:
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='gray', linestyle='-', alpha=0.5, zorder=1)

    plt.show()

# Example: 3 inputs, 4 hidden neurons, 2 outputs
draw_neural_network([3, 4, 2])
#
#
#
