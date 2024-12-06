#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include <Eigen/Sparse>
#include <Eigen/Dense>

std::pair<Eigen::VectorXd, int> preconditioned_conjugate_gradient(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& M_inv,
    int max_iter,
    double tol = 1e-6
);

#endif // PCG_SOLVER_H
