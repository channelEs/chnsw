#ifndef SEARCH_ENGINE_SIMPLE_H
#define SEARCH_ENGINE_SIMPLE_H
#pragma once
#include "search_engine.h"

class SearchEngineSimple : public SearchEngine {
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
};

#endif