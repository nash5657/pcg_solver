#include "matrix_generator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

Matrix generate_matrix_A(int n) {
    Matrix A{n, n};

    for (int i = 0; i < n; ++i) {
        if (i > 0) A.coeffRef(i, i - 1) = -1.0;  // Lower diagonal
        A.coeffRef(i, i) = 2.0;                  // Main diagonal
        if (i < n - 1) A.coeffRef(i, i + 1) = -1.0;  // Upper diagonal
    }
    A.coeffRef(n - 1, n - 1) = 1.0;  // Adjust the last diagonal element
    return A;
}

Vector generate_vector_b(int n) {
    Vector b(n, 0);
    b[0] = 1.0;
    return b;
}

Matrix read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    int rows = 0, cols = 0, nonZeros = 0;
    bool isHeader = true;
    std::vector<Eigen::Triplet<double>> triplets;

    // Parse the .mtx file format
    while (std::getline(file, line)) {
        if (line[0] == '%') continue; // Skip comments

        std::stringstream ss(line);

        if (isHeader) {
            // Read matrix dimensions and number of non-zero entries
            ss >> rows >> cols >> nonZeros;
            triplets.reserve(nonZeros); // Reserve space for triplets
            isHeader = false;
        } else {
            // Read a non-zero entry (row, col, value)
            int row, col;
            double value;
            ss >> row >> col >> value;

            // Convert to 0-based indexing for Eigen
            triplets.emplace_back(row - 1, col - 1, value);
        }
    }

    file.close();

    // Create sparse matrix
    Matrix matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

void print_matrix(const Matrix& A) {
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Matrix::InnerIterator it(A, k); it; ++it) {
            std::cout << "(" << it.row() << "," << it.col() << "): " << it.value() << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(const Vector& v) {
    for (const auto& val : v) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
