#pragma once
#include <iostream>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <cmath>
#include "utils/types.h"

class ExecutionProfiler {
public:
    void start(const std::string& tag) {
        start_times[tag] = std::chrono::high_resolution_clock::now();
        std::cout << "[PROFILER_START] " << tag << std::endl;
    }

    void stop(const std::string& tag) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = start_times[tag];
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[PROFILER_STOP] " << tag << ": " << duration << " ms (" << duration / 60000.0 << " minutes)" << std::endl;
    }

private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> start_times;
};

class ClusterEvaluator {
public:
    struct Metrics {
        double avg_intra_cluster_similarity; // High is good
        double cluster_balance_score;       // Low is good (Std Dev of sizes)
        int empty_clusters;
        std::vector<int> sizes;
    };

    static Metrics evaluate(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const ClusterResult& result
    );
};