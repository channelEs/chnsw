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
        const ExecConfig& config,
        ExecutionProfiler& profiler
    );
};