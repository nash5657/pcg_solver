#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <Eigen/Sparse>

using namespace Eigen;

// Struct to store individual matrix entries
struct Entry {
    int row, col;
    double value;
};

void readLargeMTX(const std::string& filename, int& rows, int& cols, std::vector<Entry>& entries) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return;
    }

    std::string line;
    bool isHeader = true;
    int nonZeros = 0;

    while (std::getline(file, line)) {
        if (line[0] == '%') continue;  // Skip comments

        std::istringstream iss(line);

        if (isHeader) {
            // Read matrix dimensions and number of non-zero entries
            iss >> rows >> cols >> nonZeros;
            entries.reserve(nonZeros); // Reserve space for efficiency
            isHeader = false;
        } else {
            // Read a non-zero entry
            int row, col;
            double value;
            iss >> row >> col >> value;
            entries.push_back({row - 1, col - 1, value}); // Convert to 0-based index
        }
    }

    file.close();
}

SparseMatrix<double> createSparseMatrix(int rows, int cols, const std::vector<Entry>& entries) {
    SparseMatrix<double> matrix(rows, cols);

    // Use Eigen's triplet structure for efficient sparse matrix assembly
    std::vector<Triplet<double>> triplets;
    triplets.reserve(entries.size());

    for (const auto& e : entries) {
        triplets.emplace_back(e.row, e.col, e.value);
    }

    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

int main() {
    std::string filename = "/Users/nash/Project/Roman/gyro_k.mtx";
    int rows, cols;
    std::vector<Entry> entries;

    // Read the MTX file
    readLargeMTX(filename, rows, cols, entries);

    std::cout << "Matrix dimensions: " << rows << "x" << cols << "\n";
    std::cout << "Non-zero entries: " << entries.size() << "\n";

    // Create sparse matrix
    SparseMatrix<double> sparseMatrix = createSparseMatrix(rows, cols, entries);

    std::cout << "Sparse matrix created successfully.\n";

    // // Example usage: Output a few entries
    // for (int k = 0; k < sparseMatrix.outerSize(); ++k) {
    //     for (SparseMatrix<double>::InnerIterator it(sparseMatrix, k); it; ++it) {
    //         std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
    //     }
    // }

    return 0;
}
