#include "matrix_generator.h"
#include "preconditioner.h"
#include "pcg_solver.h"
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono> // For timing

int main() {
    omp_set_num_threads(64);
    //int n = 100000; // Size of the system
    //auto A = generate_matrix_A(n);
    auto A = readMtx("gyro_k.mtx");
    int n = A.rows();
    Eigen::VectorXd b = generate_vector_b(n);

    // Create the Incomplete Cholesky preconditioner object
    Eigen::IncompleteCholesky<double> ic_preconditioner;

    // Generate the Incomplete Cholesky preconditioner
    auto precond_start_time = std::chrono::high_resolution_clock::now();
    generate_incomplete_cholesky_preconditioner(A, ic_preconditioner);
    auto precond_end_time = std::chrono::high_resolution_clock::now();
    auto precond_duration = std::chrono::duration_cast<std::chrono::milliseconds>(precond_end_time - precond_start_time);
    std::cout << "Preconditioner generation time: " << precond_duration.count() << " ms" << std::endl;

    // Solve the system using Preconditioned Conjugate Gradient
    auto solver_start_time = std::chrono::high_resolution_clock::now();
    auto [x, iterations] = preconditioned_conjugate_gradient(
        A,
        b,
        [&ic_preconditioner](const Eigen::VectorXd& r) {
            return apply_IC0_preconditioner(ic_preconditioner, r);
        },
        n,
        1e-6
    );
    auto solver_end_time = std::chrono::high_resolution_clock::now();
    auto solver_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end_time - solver_start_time);

    //std::cout << "Solution x:\n" << x.transpose() << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Solver time: " << solver_duration.count() << " ms" << std::endl;

    return 0;
}
