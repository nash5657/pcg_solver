#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>

// Type alias for convenience
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// Function declarations
Matrix generate_matrix_A(int n);
Vector generate_vector_b(int n);
void print_matrix(const Matrix& A);
void print_vector(const Vector& v);

#endif // MATRIX_GENERATOR_H
