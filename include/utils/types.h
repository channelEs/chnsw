#pragma once
#include <Eigen/Sparse>
#include <vector>

// Forward Index: just an alias for easier reading
using ForwardIndex = Eigen::SparseMatrix<float, Eigen::RowMajor>;

struct Block {
    std::vector<int> doc_ids;       // Indices of documents in this block
    Eigen::VectorXf summary;        // The "Sketch" (Max-weights for pruning)
};

struct ClusterResult {
    std::vector<int> assignments; // Which cluster each doc belongs to
    std::vector<Eigen::VectorXf> centroids; // The dense/sparse centers
};