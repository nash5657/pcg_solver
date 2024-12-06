#pragma once
#include <Eigen/Sparse>

// Generate a diagonal preconditioner (Jacobi preconditioner)
Eigen::SparseMatrix<double> generate_diagonal_preconditioner(const Eigen::SparseMatrix<double>& A);

// Generate an Incomplete Cholesky (IC(0)) preconditioner
Eigen::SparseMatrix<double> generate_incomplete_cholesky_preconditioner(const Eigen::SparseMatrix<double>& A);

// Apply the IC(0) preconditioner: Solve M y = r with M = L L^T
// Here you need a function to solve using L. You can implement forward and backward solves.
Eigen::VectorXd apply_IC0_preconditioner(const Eigen::SparseMatrix<double>& L, const Eigen::VectorXd& r);
