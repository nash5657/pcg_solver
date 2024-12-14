#include "matrix_generator.h"
#include "preconditioner.h"
#include "pcg_solver.h"
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono> // For timing

int main() {
    omp_set_num_threads(32);
    int n = 10000; // Size of the system
    auto A = generate_matrix_A(n);
    //auto A = readMtx("gyro_k.mtx");
    //int n = A.rows();
    Eigen::VectorXd b = generate_vector_b(n);

    // Create the Incomplete Cholesky preconditioner object
    Eigen::IncompleteCholesky<double> ic_preconditioner;

    // Generate the Incomplete Cholesky preconditioner
    auto precond_start_time = std::chrono::high_resolution_clock::now();
    generate_incomplete_cholesky_preconditioner(A, ic_preconditioner);
    auto precond_end_time = std::chrono::high_resolution_clock::now();
    auto precond_duration = std::chrono::duration_cast<std::chrono::milliseconds>(precond_end_time - precond_start_time);
    std::cout << "Preconditioner generation time: " << precond_duration.count() << " ms" << std::endl;

    // Solve the system using Preconditioned Conjugate Gradient with Incomplete Cholesky
    auto solver_start_time = std::chrono::high_resolution_clock::now();
    auto [x_ic, iterations_ic] = preconditioned_conjugate_gradient(
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

    std::cout << "IC0 Iterations: " << iterations_ic << std::endl;
    std::cout << "IC0 Solver time: " << solver_duration.count() << " ms" << std::endl;

    // Solve the system using Preconditioned Conjugate Gradient with Jacobi
    solver_start_time = std::chrono::high_resolution_clock::now();
    auto [x_jacobi, iterations_jacobi] = preconditioned_conjugate_gradient(
        A,
        b,
        [&A](const Eigen::VectorXd& r) {
            return apply_jacobi_preconditioner(A, r);
        },
        n,
        1e-6
    );
    solver_end_time = std::chrono::high_resolution_clock::now();
    solver_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end_time - solver_start_time);

    std::cout << "Jacobi Iterations: " << iterations_jacobi << std::endl;
    std::cout << "Jacobi Solver time: " << solver_duration.count() << " ms" << std::endl;

    // Compare the solutions
    if (x_ic.isApprox(x_jacobi, 1e-6)) {
        std::cout << "The solutions from IC0 and Jacobi are approximately equal." << std::endl;
    } else {
        std::cout << "The solutions from IC0 and Jacobi are not equal." << std::endl;
    }

    // Create the Incomplete LU preconditioner object
    Eigen::SparseLU<Eigen::SparseMatrix<double>> ilu_preconditioner;

    // Generate the Incomplete LU preconditioner
    precond_start_time = std::chrono::high_resolution_clock::now();
    generate_incomplete_lu_preconditioner(A, ilu_preconditioner);
    precond_end_time = std::chrono::high_resolution_clock::now();
    precond_duration = std::chrono::duration_cast<std::chrono::milliseconds>(precond_end_time - precond_start_time);
    std::cout << "ILU Preconditioner generation time: " << precond_duration.count() << " ms" << std::endl;

    // Solve the system using Preconditioned Conjugate Gradient with ILU
    solver_start_time = std::chrono::high_resolution_clock::now();
    auto [x_ilu, iterations_ilu] = preconditioned_conjugate_gradient(
        A,
        b,
        [&ilu_preconditioner](const Eigen::VectorXd& r) {
            return apply_ILU_preconditioner(ilu_preconditioner, r);
        },
        n,
        1e-6
    );
    solver_end_time = std::chrono::high_resolution_clock::now();
    solver_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solver_end_time - solver_start_time);

    std::cout << "ILU Iterations: " << iterations_ilu << std::endl;
    std::cout << "ILU Solver time: " << solver_duration.count() << " ms" << std::endl;

    // Compare the solutions
    if (x_ic.isApprox(x_ilu, 1e-6)) {
        std::cout << "The solutions from IC0 and ILU are approximately equal." << std::endl;
    } else {
        std::cout << "The solutions from IC0 and ILU are not equal." << std::endl;
    }

    if (x_jacobi.isApprox(x_ilu, 1e-6)) {
        std::cout << "The solutions from Jacobi and ILU are approximately equal." << std::endl;
    } else {
        std::cout << "The solutions from Jacobi and ILU are not equal." << std::endl;
    }

    return 0;
}