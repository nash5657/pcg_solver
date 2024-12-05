#include "preconditioner.h"
#include <stdexcept>
#include <cmath>

// Generate a diagonal preconditioner (Jacobi preconditioner)
Matrix generate_diagonal_preconditioner(const Matrix& A) {
    int n = A.size();
    Matrix M_inv(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) {
        if (A[i][i] != 0.0) {
            M_inv[i][i] = 1.0 / A[i][i];
        }
    }
    return M_inv;
}

// Generate an incomplete Cholesky preconditioner
Matrix generate_incomplete_cholesky_preconditioner(const Matrix& A) {
    int n = A.size();
    Matrix L(n, Vector(n, 0.0));

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * L[i][j];
        }

        if (A[i][i] - sum < 0) {
            throw std::runtime_error("Matrix is not positive definite");
        }

        L[i][i] = std::sqrt(A[i][i] - sum);

        for (int j = i + 1; j < n; ++j) {
            sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[j][k] * L[i][k];
            }
            L[j][i] = (A[j][i] - sum) / L[i][i];
        }
    }

    return L;
}


// Apply preconditioner (M_inv * r)
void apply_preconditioner(const Matrix& L, const Vector& r, Vector& result) {
    int n = L.size();
    Vector y(n, 0.0);

    // Step 1: Solve L * y = r (forward substitution)
    for (int i = 0; i < n; ++i) {
        y[i] = r[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }

    // Step 2: Solve L^T * result = y (backward substitution)
    result.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        result[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            result[i] -= L[j][i] * result[j];
        }
        result[i] /= L[i][i];
    }
}

