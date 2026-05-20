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
    static void evaluate(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const ClusterResult& result
    ) 
    {
        int total_docs = data.rows();
        int n_clusters = result.centroids.size();

        double total_sim = 0;
        for (int i = 0; i < total_docs; ++i) {
            int c = result.assignments[i];
            // Dot product between doc and its assigned centroid
            total_sim += data.row(i).dot(result.centroids[c]);
        }
        double avg_intra_cluster_similarity = total_sim / total_docs;

        std::vector<int> counts(n_clusters, 0);
        for (int a : result.assignments) counts[a]++;
        
        // median cluster size
        std::vector<int> sorted_counts = counts;
        std::sort(sorted_counts.begin(), sorted_counts.end());
        int median_cluster_size = sorted_counts[n_clusters / 2];

        int largest_cluster = *std::max_element(counts.begin(), counts.end());
        int smallest_cluster = *std::min_element(counts.begin(), counts.end());
        int empty_clusters = std::count(counts.begin(), counts.end(), 0);

        double avg_cluster_size = static_cast<double>(total_docs) / n_clusters;

        double size_deviation = 0.0;
        for (int size : counts) {
            size_deviation += std::pow(size - avg_cluster_size, 2);
        }
        std::cout << "\n--- Evaluate clustering (relating docs information) ---" << std::endl;
        size_deviation = std::sqrt(size_deviation / n_clusters);
        std::cout << "Objective Average Cluster Size: " << avg_cluster_size << "\n";
        std::cout << "Median Cluster Size: " << median_cluster_size << "\n";
        std::cout << "Largest Cluster Size: " << largest_cluster << "\n";
        std::cout << "Smallest Cluster Size: " << smallest_cluster << "\n";
        std::cout << "Empty Clusters: " << empty_clusters << "\n";
        std::cout << "Cluster Size Deviation: " << size_deviation << "\n";  
        std::cout << "Avg Similarity (Docs and its cluster assignments with dot product): " << avg_intra_cluster_similarity << std::endl;
    }
};