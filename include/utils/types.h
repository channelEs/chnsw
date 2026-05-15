#pragma once
#include <Eigen/Sparse>
#include <vector>

// Forward Index: just an alias for easier reading
using ForwardIndex = Eigen::SparseMatrix<float, Eigen::RowMajor>;

struct Block {
    std::vector<int> doc_ids;       // Indices of documents in this block
    Eigen::VectorXf summary;        // The "Sketch" (Max-weights for pruning)
};