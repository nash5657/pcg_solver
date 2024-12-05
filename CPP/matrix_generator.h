#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>
#include <string>
#include <Eigen/Sparse>

// Type alias for convenience
// using Vector = std::vector<double>;
// using Matrix = std::vector<std::vector<double>>;
using Vector = Eigen::VectorXd;
using Matrix = Eigen::SparseMatrix<double>;

// Function declarations
Matrix generate_matrix_A(int n);
Vector generate_vector_b(int n);
Matrix read_matrix_from_file(const std::string& filename); // New function
void print_matrix(const Matrix& A);
void print_vector(const Vector& v);

#endif // MATRIX_GENERATOR_H
