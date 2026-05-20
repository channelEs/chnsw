#pragma once
#include <Eigen/Sparse>
#include <vector>
#include "utils/types.h"
#include "utils/metrics.h"

class ClusteringEngine {
public:
    // Performs Spherical K-means
    static ClusterResult run(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data, 
        int num_clusters, 
        ExecutionProfiler& profiler,
        int max_iterations = 10
    );
};