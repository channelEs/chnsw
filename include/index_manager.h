#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

/**
 * @brief Represents a block of documents within a single concept's inverted list.
 * All documents in this block belong to the same cluster.
 */
struct InvertedBlock {
    int cluster_id;
    std::vector<int> doc_ids;
    std::vector<float> weights;
};

class IndexManager {
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
    );

    /**
     * @brief Builds the partitioned inverted index.
     * The outer vector is indexed by Concept ID (0 to 30,521).
     * The inner vector contains blocks of documents, grouped by cluster.
     */
    std::vector<std::vector<InvertedBlock>> buildInvertedIndex(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const std::vector<int>& assignments,
        int num_clusters
    );
};