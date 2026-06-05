#ifndef INDEX_MANAGER_OPTIMIZED_H
#define INDEX_MANAGER_OPTIMIZED_H
#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include "utils/types.h"
#include "index_manager.h"

class IndexManagerOptimized : public IndexManager {
public:
    /**
     * @brief Computes a "Summary Vector" for each cluster.
     * The Summary Vector stores the maximum weight for every dimension found 
     * across all documents in that cluster.
     */
    std::vector<Eigen::VectorXf> computeSummaryVectors(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const std::vector<int>& assignments,
        int num_clusters
    ) override;

    /**
     * @brief Builds the partitioned inverted index.
     * The outer vector is indexed by Concept ID (0 to 30,521).
     * The inner vector contains blocks of documents, grouped by cluster.
     */
    std::vector<std::vector<InvertedBlock>> buildInvertedIndex(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const std::vector<int>& assignments,
        int num_clusters,
        const struct ExecConfig& config,
        const std::vector<Eigen::VectorXf>& summaries
    ) override;
};

#endif