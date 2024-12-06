#include "preconditioner.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// // Generate a diagonal preconditioner (Jacobi preconditioner)
// Eigen::SparseMatrix<double> generate_diagonal_preconditioner(const Eigen::SparseMatrix<double>& A) {
//     Eigen::SparseMatrix<double> M_inv(A.rows(), A.cols());
//     M_inv.reserve(Eigen::VectorXi::Constant(A.rows(), 1)); // Reserve one non-zero per row
//     for (int k = 0; k < A.outerSize(); ++k) {
//         for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
//             if (it.row() == it.col() && it.value() != 0.0) {
//                 M_inv.insert(it.row(), it.col()) = 1.0 / it.value();
//             }
//         }
//     }
//     M_inv.makeCompressed();
//     return M_inv;
// }

Eigen::VectorXd apply_jacobi_preconditioner(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& r) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix A must be square.");
    }

    Eigen::VectorXd diag = A.diagonal(); // Extract diagonal elements of A
    if (diag.minCoeff() == 0) { 
        throw std::runtime_error("Matrix A contains zero diagonal elements, making Jacobi preconditioning invalid.");
    }

    Eigen::VectorXd result = r.cwiseQuotient(diag); // Element-wise division of r by diag
    return result;
}


// Generate an Incomplete Cholesky (IC(0)) preconditioner
// This code computes L such that L L^T approximates A.
// We do not add any fill-in beyond the pattern of A.
Eigen::SparseMatrix<double> generate_incomplete_cholesky_preconditioner(const Eigen::SparseMatrix<double>& A) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("Matrix must be square for IC(0).");
    }

    int n = A.rows();

    // We assume A is SPD. No checks are made here.

    // We'll compute L in place. L will have the same pattern as A but we'll store only the lower part.
    // Extract the lower-triangular structure of A (including diagonal):
    Eigen::SparseMatrix<double> L = A.triangularView<Eigen::Lower>();

    // We now perform the IC(0) factorization:
    // For each row i:
    //    L(i,i) = sqrt(L(i,i) - sum(L(i,k)^2 for k < i))
    // For each non-diagonal j < i:
    //    L(i,j) = (L(i,j) - sum(L(i,k)*L(j,k) for k < j)) / L(j,j)
    //
    // If any diagonal becomes non-positive, that means we have a numerical issue or A is not SPD.
    // For IC(0), we skip any entries not in A’s pattern (no fill-in).

    L.makeCompressed();
    // Create a pointer-based structure to access L's values easily
    const int* Lp = L.outerIndexPtr();
    const int* Li = L.innerIndexPtr();
    double* Lx = const_cast<double*>(L.valuePtr());

    // We'll do a standard Cholesky loop, but only for entries in L.
    for (int i = 0; i < n; ++i) {
        // First compute L(i,i)
        double sum_diag = 0.0;
        // Traverse current row i for columns < i
        for (int idx = Lp[i]; idx < Lp[i+1]; ++idx) {
            int j = Li[idx];
            if (j >= i) break; // Since it's lower-triangular, once j==i we've reached diagonal
            double Lij = Lx[idx];
            sum_diag += Lij*Lij;
        }

        // Find diagonal index in row i
        int diag_idx = -1;
        for (int idx = Lp[i]; idx < Lp[i+1]; ++idx) {
            if (Li[idx] == i) {
                diag_idx = idx;
                break;
            }
        }

        if (diag_idx == -1) {
            throw std::runtime_error("No diagonal element found in row " + std::to_string(i));
        }

        double val_ii = Lx[diag_idx] - sum_diag;
        if (val_ii <= 0.0) {
            throw std::runtime_error("Incomplete Cholesky failed: non-positive pivot encountered at row " + std::to_string(i));
        }

        double diag_val = std::sqrt(val_ii);
        Lx[diag_idx] = diag_val;

        // Now compute off-diagonal elements in row i
        for (int idx = Lp[i]; idx < Lp[i+1]; ++idx) {
            int j = Li[idx];
            if (j >= i) break; // Only process strictly lower elements

            // Compute L(i,j) = (L(i,j) - sum_k L(i,k)*L(j,k))/L(j,j)
            double sum_off = 0.0;

            // To compute sum_k L(i,k)*L(j,k), we need to traverse row i and j up to k < j
            // A simple but less efficient approach is to do a merge-like traversal:
            int p_i = Lp[i]; 
            int p_j = Lp[j]; 
            int end_i = Lp[i+1]; 
            int end_j = Lp[j+1]; 

            while (p_i < end_i && p_j < end_j) {
                int col_i = Li[p_i]; 
                int col_j = Li[p_j];
                if (col_i < col_j) {
                    p_i++;
                } else if (col_j < col_i) {
                    p_j++;
                } else {
                    // col_i == col_j == k
                    if (col_i < j) { 
                        sum_off += Lx[p_i]*Lx[p_j];
                    }
                    p_i++;
                    p_j++;
                }
            }

            // Find the diagonal of L(j,j)
            double Ljj = 0.0;
            {
                int diag_j_idx = -1;
                for (int idx_j = Lp[j]; idx_j < Lp[j+1]; ++idx_j) {
                    if (Li[idx_j] == j) {
                        diag_j_idx = idx_j; 
                        break;
                    }
                }
                if (diag_j_idx == -1) {
                    throw std::runtime_error("No diagonal element found in row j=" + std::to_string(j));
                }
                Ljj = Lx[diag_j_idx];
            }

            double new_val = (Lx[idx] - sum_off) / Ljj;
            Lx[idx] = new_val;
        }
    }

    return L;
}

// Apply the IC(0) preconditioner: we need to solve M y = r with M = L L^T.
// This involves forward solve L z = r, and then backward solve L^T y = z.
Eigen::VectorXd apply_IC0_preconditioner(const Eigen::SparseMatrix<double>& L, const Eigen::VectorXd& r) {
if (L.rows() != L.cols() || L.rows() != r.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    int n = L.rows();
    Eigen::VectorXd y(n);
    Eigen::VectorXd result(n);
    y.setZero();
    result.setZero();

    // Forward substitution: L * y = r
    for (int i = 0; i < n; ++i) {
        double sum = r[i];
        for (int j = 0; j < i; ++j) {
            double val = L.coeff(i, j);  // zero if not present
            if (val != 0.0) {
                sum -= val * y[j];
            }
        }
        double diag = L.coeff(i, i);
        if (diag == 0.0) {
            throw std::runtime_error("Zero diagonal element encountered in L.");
        }
        y[i] = sum / diag;
    }

    // Backward substitution: L^T * result = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            double val = L.coeff(j, i);
            if (val != 0.0) {
                sum -= val * result[j];
            }
        }
        double diag = L.coeff(i, i);
        if (diag == 0.0) {
            throw std::runtime_error("Zero diagonal element encountered in L.");
        }
        result[i] = sum / diag;
    }

    return result;
}
