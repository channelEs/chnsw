#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H
#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include "index_manager.h"

class SearchEngine {
public:
    virtual ~SearchEngine() = default;
    
    /**
     * @brief Executes a Top-K search for a single query using Summary Vector block pruning.
     * @param train The forward index (collection of sparse document vectors).
     * @param inverted_index The partitioned inverted index grouped by clusters.
     * @param summary_vectors The max-weight bounding vectors for each cluster.
     * @param query_matrix The matrix containing the sparse queries.
     * @param q_idx The index of the specific query to run.
     * @param k The number of results to return (e.g., 30).
     * @param config Global execution configuration.
     * @return std::vector<std::pair<float, int>> Top-K results as pairs of (score, doc_id) sorted descending.
     */
    virtual std::vector<std::pair<float, int>> search(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& train,
        const std::vector<std::vector<InvertedBlock>>& inverted_index,
        const std::vector<Eigen::VectorXf>& summary_vectors,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& query_matrix,
        int q_idx,
        int k,
        const struct ExecConfig& config
    ) = 0;
};

#endif