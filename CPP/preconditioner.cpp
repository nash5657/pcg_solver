#include "preconditioner.h"
#include <stdexcept>

// Generate a diagonal preconditioner (Jacobi preconditioner)
Matrix generate_diagonal_preconditioner(const Matrix& A) {
    int n = A.size();
    Matrix M_inv(n, n);
    for (int i = 0; i < n; ++i) {
        if (A.coeff(i, i) != 0.0) {
            M_inv.insert(i, i) = 1.0 / A.coeff(i, i);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (A.coeff(i, i) != 0.0) {
            M_inv.coeffRef(i, i) = 1.0 / A.coeff(i, i);
        }
    }
    return M_inv;
}

// Generate an incomplete Cholesky preconditioner
Matrix generate_incomplete_cholesky_preconditioner(const Matrix& A) {
    int n = A.size();
    
    // Validate matrix is square
    for (int i = 0; i < n; ++i) {
        if (A.cols() != n) {
            throw std::invalid_argument("Matrix must be square");
        }
    }

    // Create lower triangular matrix for Cholesky factorization
    Matrix L(n, n);

    // Incomplete Cholesky Factorization
    for (int i = 0; i < n; ++i) {
        // Compute diagonal element
        double diag_sum = A.coeff(i, i);
        
        // Subtract previous contributions
        for (int k = 0; k < i; ++k) {
            diag_sum -= L.coeff(i, k) * L.coeff(i, k);
        }

        // Check positive definiteness
        if (diag_sum <= 0) {
            throw std::runtime_error("Matrix is not positive definite");
        }

        L.coeffRef(i, i) = std::sqrt(diag_sum);

        // Compute lower triangular elements
        for (int j = i + 1; j < n; ++j) {
            double sum = A.coeff(j, i);
            
            // Subtract previous contributions
            for (int k = 0; k < i; ++k) {
                sum -= L.coeff(i, k) * L.coeff(j, k);
            }

            // Compute L[j][i]
            L.coeffRef(j, i) = sum / L.coeff(i, i);
        }
    }

    return L;
}


// Apply preconditioner (M_inv * r)
void apply_preconditioner(const Matrix& L, const Vector& r, Vector& result) {
    int n = L.size();
    Vector y(n, 0);

    // Step 1: Solve L * y = r (forward substitution)
    for (int i = 0; i < n; ++i) {
        y[i] = r[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L.coeff(i, j) * y[j];
        }
        y[i] /= L.coeff(i, i);
    }

    // Step 2: Solve L^T * result = y (backward substitution)
    result.setZero();
    for (int i = n - 1; i >= 0; --i) {
        result[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            result[i] -= L.coeff(j, i) * result[j];
        }
        result[i] /= L.coeff(i, i);
    }
}
