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

    /**
     * @brief Prints average debug statistics collected during search (e.g., blocks entered/skipped, docs examined/popped).
     * Implementations can maintain internal counters to compute these averages across multiple queries.
     */
    virtual void printAvgDebugStats() const = 0;

    /**
     * @brief Retrieves average debug statistics collected during search.
     * @param avg_blocks_entered Average number of blocks entered per query.
     * @param avg_blocks_skipped Average number of blocks skipped per query.
     * @param avg_docs_examined Average number of documents examined per query.
     * @param avg_docs_popped Average number of documents popped from the heap per query.
     */
    virtual void getAvgDebugStats(double& avg_blocks_entered, double& avg_blocks_skipped, double& avg_docs_examined, double& avg_docs_popped) const = 0;

    protected:
        long long total_blocks_entered = 0;
        long long total_blocks_skipped = 0;
        long long total_docs_examined = 0;
        long long total_docs_popped = 0;
        long long num_queries_run = 0;
};

#endif