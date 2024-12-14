#include "preconditioner.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

Eigen::VectorXd apply_jacobi_preconditioner(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& r) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix A must be square.");
    }

    Eigen::VectorXd diag = A.diagonal();
    if (diag.minCoeff() == 0) { 
        throw std::runtime_error("Matrix A contains zero diagonal elements, making Jacobi preconditioning invalid.");
    }

    return r.cwiseQuotient(diag); // Element-wise division
}

void generate_incomplete_cholesky_preconditioner(const Eigen::SparseMatrix<double>& A, Eigen::IncompleteCholesky<double>& ic) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("Matrix must be square for IC(0).");
    }

    // Compute the IC(0) preconditioner in-place
    ic.compute(A);

    if (ic.info() != Eigen::Success) {
        throw std::runtime_error("Incomplete Cholesky factorization failed!");
    }
}

Eigen::VectorXd apply_IC0_preconditioner(const Eigen::IncompleteCholesky<double>& ic, const Eigen::VectorXd& r) {
    return ic.solve(r);
}
// Incomplete LU preconditioner
void generate_incomplete_lu_preconditioner(const Eigen::SparseMatrix<double>& A, Eigen::SparseLU<Eigen::SparseMatrix<double>>& ilu) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("Matrix must be square for ILU.");
    }

    // Compute the ILU preconditioner
    ilu.analyzePattern(A);
    ilu.factorize(A);

    if (ilu.info() != Eigen::Success) {
        throw std::runtime_error("Incomplete LU factorization failed!");
    }
}

Eigen::VectorXd apply_ILU_preconditioner(const Eigen::SparseLU<Eigen::SparseMatrix<double>>& ilu, const Eigen::VectorXd& r) {
    return ilu.solve(r);
}
