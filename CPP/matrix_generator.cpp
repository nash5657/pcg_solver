#include "matrix_generator.h"
#include <iostream>

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
