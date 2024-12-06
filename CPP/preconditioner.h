#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <vector>

// Type alias for convenience
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// Function declarations
// Matrix generate_diagonal_preconditioner(const Matrix& A);
Matrix generate_incomplete_cholesky_preconditioner(const Matrix& A);
void apply_preconditioner(const Matrix& M_inv, const Vector& r, Vector& result);
void apply_jacobi_preconditioner(const Matrix& A, const Vector& r, Vector& result);

#endif // PRECONDITIONER_H
