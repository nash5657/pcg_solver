#include "matrix_generator.h"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::SparseMatrix<double> generate_matrix_A(int n) {
    std::vector<Eigen::Triplet<double>> triplets;

    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0); // Lower diagonal
        }

        if (i == n - 1) {
            // For the last element A(n-1, n-1) = 1
            triplets.emplace_back(i, i, 1.0);
        } else {
            // For all other diagonal elements A(i, i) = 2
            triplets.emplace_back(i, i, 2.0);
        }

        if (i < n - 1) {
            triplets.emplace_back(i, i + 1, -1.0); // Upper diagonal
        }
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}


Eigen::VectorXd generate_vector_b(int n) {
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    b[0] = 1.0; // Example RHS vector with 1 in the first position
    return b;
}
