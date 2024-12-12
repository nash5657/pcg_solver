#include "pcg_solver.h"
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <functional>

std::pair<Eigen::VectorXd, int> preconditioned_conjugate_gradient(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> apply_preconditioner,
    int max_iter,
    double tol
) {
    int n = b.size();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n); // Initial guess x0 = 0
    Eigen::VectorXd r = b - A * x;                // Initial residual
    Eigen::VectorXd y = apply_preconditioner(r);  // Preconditioned residual
    Eigen::VectorXd p = y;                        // Search direction

    double mu_prev = r.dot(y);
    int k = 0;

    // Parallelize the matrix-vector product
    for (k = 1; k <= max_iter; ++k) {
        Eigen::VectorXd z(n);
        #pragma omp parallel for
        for (int i = 0; i < A.outerSize(); ++i) {
            z[i] = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                z[i] += it.value() * p[it.index()];
            }
        }

        double nu = mu_prev / p.dot(z);           // Step size

        x += nu * p;                              // Update solution
        r -= nu * z;                              // Update residual

        if (r.norm() < tol) {                     // Convergence check
            break;
        }

        y = apply_preconditioner(r);              // Apply preconditioner
        
        double mu = 0.0;

        // Parallelize the dot product
        #pragma omp parallel for reduction(+:mu)
        for (int i = 0; i < n; ++i) {
            mu += r[i] * y[i];
        }

        double beta = mu / mu_prev;               // Compute new beta
        p = y + beta * p;                         // Update search direction
        mu_prev = mu;
    }

    return {x, k};
}
