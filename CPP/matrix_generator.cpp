#include "matrix_generator.h"
#include <iostream>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

Matrix loadMatrix(const std::string& filename) {
    std::ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file " + filename);
    }

    // Skip comment lines (%)
    // Find the line with matrix dimensions
    std::string line;
    while (true) {
        if (!std::getline(infile, line)) {
            throw std::runtime_error("File ended before matrix dimensions were found.");
        }
        if (!line.empty() && line[0] != '%') {
            // This line should contain: numRows numCols numNonzeros
            break;
        }
    }

    int numRows, numCols, numNonzeros;
    {
        std::stringstream ss(line);
        ss >> numRows >> numCols >> numNonzeros;
        if (ss.fail()) {
            throw std::runtime_error("Error parsing matrix dimension line.");
        }
    }

    // Initialize the matrix with zeros
    std::vector<std::vector<double>> matrix(numRows, std::vector<double>(numCols, 0.0));

    // Read the nonzero entries
    // Format: i j value  (1-based indexing)
    for (int k = 0; k < numNonzeros; ++k) {
        if (!std::getline(infile, line)) {
            throw std::runtime_error("Unexpected end of file while reading entries.");
        }
        std::stringstream entryStream(line);
        int i, j;
        double value;
        entryStream >> i >> j >> value;
        if (entryStream.fail()) {
            throw std::runtime_error("Error parsing a nonzero entry line.");
        }

        // Convert to 0-based indexing
        i -= 1;
        j -= 1;

        // Place the value
        matrix[i][j] = value;
        // Since the matrix is symmetric, mirror the value if i != j
        if (i != j) {
            matrix[j][i] = value;
        }
    }

    return matrix;
}

Matrix generate_matrix_A(int n) {
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        if (i > 0) A[i][i - 1] = -1.0;  // Lower diagonal
        A[i][i] = 2.0;                  // Main diagonal
        if (i < n - 1) A[i][i + 1] = -1.0;  // Upper diagonal
    }
    A[n - 1][n - 1] = 1.0;  // Adjust the last diagonal element
    return A;
}

Vector generate_vector_b(int n) {
    Vector b(n, 0.0);
    b[0] = 1.0;
    return b;
}

void print_matrix(const Matrix& A) {
    for (const auto& row : A) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(const Vector& v) {
    for (double val : v) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
