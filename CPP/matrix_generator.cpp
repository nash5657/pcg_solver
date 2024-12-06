#include "matrix_generator.h"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Sparse>

Eigen::SparseMatrix<double> readMtx(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    // Skip any header lines starting with '%'
    while (true) {
        if (!std::getline(infile, line)) {
            throw std::runtime_error("Unexpected end of file while reading header.");
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] != '%') {
            // This should be the size line
            break;
        }
    }

    int rows, cols, nnz;
    {
        std::stringstream ss(line);
        ss >> rows >> cols >> nnz;
        if (ss.fail()) {
            throw std::runtime_error("Error parsing matrix dimensions from line: " + line);
        }
    }

    // Reserve space for triplets
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    // Read the non-zero entries
    // Format: i j value  (1-based indices)
    for (int k = 0; k < nnz; ++k) {
        if (!std::getline(infile, line)) {
            throw std::runtime_error("Unexpected end of file while reading entries.");
        }
        int i, j;
        double val;
        {
            std::stringstream ss(line);
            ss >> i >> j >> val;
            if (ss.fail()) {
                throw std::runtime_error("Error parsing line: " + line);
            }
        }

        // Convert to 0-based indexing
        i -= 1;
        j -= 1;
        triplets.push_back(Eigen::Triplet<double>(i, j, val));
    }

    // Now create the sparse matrix
    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());

    return mat;
}

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
