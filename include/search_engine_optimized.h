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
        float heap_factor = 1.0f,
        int max_docs_to_visit = 10000
    ) override;
};

#endif