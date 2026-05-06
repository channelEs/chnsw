#pragma once
#include <Eigen/Sparse>
#include <vector>

class ClusteringEngine {
public:
    struct ClusterResult {
        std::vector<int> assignments; // Which cluster each doc belongs to
        std::vector<Eigen::VectorXf> centroids; // The dense/sparse centers
    };

    // Performs Spherical K-means
    static ClusterResult run(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data, 
        int num_clusters, 
        int max_iterations = 10
    );
};