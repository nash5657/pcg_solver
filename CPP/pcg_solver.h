#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include <vector>
#include <utility>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

std::pair<Vector, int> preconditioned_conjugate_gradient(
    const Matrix& A,
    const Vector& b,
    const Matrix& M_inv,
    int max_iter,
    double tol = 1e-6
);

#endif // PCG_SOLVER_H
