import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# Function to create matrix A
def create_matrix_A(n):
    diagonals = [
        -1 * np.ones(n - 1),  # Lower diagonal
        2 * np.ones(n),      # Main diagonal
        -1 * np.ones(n - 1)  # Upper diagonal
    ]
    return diags(diagonals, offsets=[-1, 0, 1], format='csr')

# Custom Conjugate Gradient Implementation
def conjugate_gradient(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)  # Initial guess x0
    r = b - A @ x  # Residual
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        
        if np.sqrt(rs_new) < tol:
            print(f"Converged in {i + 1} iterations.")
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x

# Main experiment
def run_experiment(n):
    A = create_matrix_A(n)
    b = np.zeros(n)
    b[0] = 1  # As per problem definition
    
    # Solve using custom CG
    print(f"Running CG for n={n}")
    x_custom = conjugate_gradient(A, b)
    
    # Validate with scipy's cg solver
    x_scipy, info = cg(A, b, atol=1e-6)  # Fixed here
    print(f"Scipy CG Convergence Info: {info}")
    
    return x_custom, x_scipy


# Run for different matrix sizes
for n in [10, 50, 100]:
    x_custom, x_scipy = run_experiment(n)
    print(f"Custom Solution (n={n}): {x_custom}")
    print(f"Scipy Solution (n={n}): {x_scipy}")
