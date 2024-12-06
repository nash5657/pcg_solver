#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "matrix_generator.h"
#include "preconditioner.h"
#include "pcg_solver.h"

int main() {
    int n = 10; // Size of the system

    auto A = generate_matrix_A(n);
    Eigen::VectorXd b = generate_vector_b(n);

    // Use Incomplete Cholesky preconditioner
    //Eigen::SparseMatrix<double> M_inv = generate_incomplete_cholesky_preconditioner(A);

    // Use Jacobi preconditioner
    Eigen::SparseMatrix<double> M_inv = generate_diagonal_preconditioner(A);

    auto [x, iterations] = preconditioned_conjugate_gradient(A, b, M_inv, n, 1e-6);

    std::cout << "Solution x:\n" << x.transpose() << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    return 0;
}
