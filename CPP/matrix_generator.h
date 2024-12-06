#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::SparseMatrix<double> generate_matrix_A(int n);
Eigen::VectorXd generate_vector_b(int n);

#endif // MATRIX_GENERATOR_H
