#pragma once
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

Eigen::VectorXd apply_jacobi_preconditioner(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& r);
void generate_incomplete_cholesky_preconditioner(const Eigen::SparseMatrix<double>& A, Eigen::IncompleteCholesky<double>& ic);
Eigen::VectorXd apply_IC0_preconditioner(const Eigen::IncompleteCholesky<double>& ic, const Eigen::VectorXd& r);
