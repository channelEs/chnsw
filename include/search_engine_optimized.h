#ifndef SEARCH_ENGINE_OPTIMIZED_H
#define SEARCH_ENGINE_OPTIMIZED_H
#pragma once

#include "search_engine.h"

class SearchEngineOptimized : public SearchEngine {
public:    
    std::vector<std::pair<float, int>> search(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& train,
        const std::vector<std::vector<InvertedBlock>>& inverted_index,
        const std::vector<Eigen::VectorXf>& summary_vectors,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& query_matrix,
        int q_idx,
        int k,
        const struct ExecConfig& config
    ) override;

    // Aggregated debug counters across all queries (averaged and printed by caller)
    void printAvgDebugStats() const override;
    void getAvgDebugStats(double& avg_blocks_entered, double& avg_blocks_skipped, double& avg_docs_examined, double& avg_docs_popped) const override;
};

#endif