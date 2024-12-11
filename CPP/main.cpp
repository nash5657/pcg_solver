#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "matrix_generator.h"
#include "preconditioner.h"
#include "pcg_solver.h"
#include <omp.h>
#include <chrono> // For timing

int main() {
    omp_set_num_threads(16);
    int n = 100; // Size of the system
    auto A = generate_matrix_A(n);

    //auto A = readMtx("gyro_k.mtx");
    //int n = A.rows();

    Eigen::VectorXd b = generate_vector_b(n);

    // Use Incomplete Cholesky preconditioner
    // Measure time for the preconditioner generation (if relevant)
    auto precond_start_time = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> L = generate_incomplete_cholesky_preconditioner(A);
    auto precond_end_time = std::chrono::high_resolution_clock::now();
    auto precond_duration = std::chrono::duration_cast<std::chrono::milliseconds>(precond_end_time - precond_start_time);
    std::cout << "Preconditioner generation time: " << precond_duration.count() << " ms" << std::endl;

// Measure time for the solver
    auto solver_start_time = std::chrono::high_resolution_clock::now();
    auto [x, iterations] = preconditioned_conjugate_gradient(
        A, b, [&L](auto& r) { return apply_IC0_preconditioner(L, r); }, n, 1e-6);
    auto solver_end_time = std::chrono::high_resolution_clock::now();
    auto solver_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end_time - solver_start_time);

    // Use Jacobi preconditioner
    //auto [x, iterations] = preconditioned_conjugate_gradient(A, b, [&A](auto& r) { return apply_jacobi_preconditioner(A, r);}, n, 1e-6);


    std::cout << "Solution x:\n" << x.transpose() << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Solver time: " << solver_duration.count() << " ms" << std::endl;


    return 0;
}
