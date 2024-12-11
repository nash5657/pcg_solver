#include "pcg_solver.h"
#include <iostream>

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

    for (k = 1; k <= max_iter; ++k) {
        Eigen::VectorXd z = A * p;                // Matrix-vector  /focus here
        double nu = mu_prev / p.dot(z);           // Step size

        x += nu * p;                              // Update solution
        r -= nu * z;                              // Update residual

        if (r.norm() < tol) {                     // Convergence check
            break;
        }

        y = apply_preconditioner(r);              // Apply preconditioner
        double mu = r.dot(y);                     // Updated scalar for direction   // focus here
        double beta = mu / mu_prev;               // Compute new beta
        p = y + beta * p;                         // Update search direction
        mu_prev = mu;
    }

    return {x, k};
}
