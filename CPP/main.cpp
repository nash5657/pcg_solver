#include <iostream>
#include <cmath>
#include "matrix_generator.h"
#include "preconditioner.h"
#include "pcg_solver.h"

void matvec(const Matrix& A, const Vector& x, Vector& result) {
    int n = A.size();
    result.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
}

double dot(const Vector& a, const Vector& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::pair<Vector, int> preconditioned_conjugate_gradient(
    const Matrix& A,
    const Vector& b,
    std::function<void(const Vector&, Vector&)> preconditioner,
    int max_iter,
    double tol
) {
    int n = b.size();
    Vector x(n, 0.0);  // Initial guess x0 = 0
    Vector r(n), y(n), p(n), z(n);

    matvec(A, x, r);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }

    preconditioner(r, y);
    p = y;

    double mu_prev = dot(r, y);

    int k = 0;
    for (k = 1; k <= max_iter; ++k) {
        matvec(A, p, z);
        double nu = mu_prev / dot(p, z);

        for (int i = 0; i < n; ++i) {
            x[i] += nu * p[i];
        }

        for (int i = 0; i < n; ++i) {
            r[i] -= nu * z[i];
        }

        if (std::sqrt(dot(r, r)) < tol) {
            break;
        }

        preconditioner(r, y);
        double mu = dot(r, y);
        double beta = mu / mu_prev;

        for (int i = 0; i < n; ++i) {
            p[i] = y[i] + beta * p[i];
        }

        mu_prev = mu;
    }

    return {x, k};
}

int main() {

    auto A = loadMatrix("gyro_k.mtx");
    int n = A.size();
    // std::cout << "size of A: " << n << std::endl;
    // for(int i = 0; i < 10; i++){
    //     for(int j = 0; j < 10; j++){
    //         std::cout << A[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // return 0;
    // int n = 1000; // Size of the system
    // auto A = generate_matrix_A(n);
    auto b = generate_vector_b(n);

    //Use Jacobi preconditioner
    // Matrix M_inv = generate_diagonal_preconditioner(A);
    auto result = preconditioned_conjugate_gradient(A, b, [&A](auto&r, auto&y) { apply_jacobi_preconditioner(A, r, y);}, n, 1e-6);

    // Use Incomplete Cholesky preconditioner
    // Matrix L = generate_incomplete_cholesky_preconditioner(A);
    // auto result = preconditioned_conjugate_gradient(A, b, [&L](auto&r, auto&y) {apply_preconditioner(L, r, y);}, n, 1e-6);
    auto x = result.first;
    auto iterations = result.second;

    std::cout << "Solution x:\n";
    for (auto xi : x) {
        std::cout << xi << " ";
    }
    std::cout << "\nIterations: " << iterations << std::endl;

    return 0;
}
