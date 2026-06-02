#include "index_manager.h"
#include <iostream>
#include <algorithm>
#include <numeric>

std::vector<Eigen::VectorXf> IndexManager::computeSummaryVectors(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const std::vector<int>& assignments,
    int num_clusters
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    // Initialize dense vectors to store the max weights for each cluster
    std::vector<Eigen::VectorXf> summaries(num_clusters, Eigen::VectorXf::Zero(n_dims));

    std::cout << "[INDEXING] Computing Summary Vectors (Sketches) for " << num_clusters << " clusters..." << std::endl;

    for (int i = 0; i < n_docs; ++i) {
        int cluster_id = assignments[i];
        
        // Use Eigen's InnerIterator to touch only non-zero values (High Efficiency)
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
            int term_idx = it.index();
            float weight = it.value();

            // We want the 'ceiling' (max weight) for this concept in this cluster
            if (weight > summaries[cluster_id][term_idx]) {
                summaries[cluster_id][term_idx] = weight;
            }
        }
    }

    return summaries;
}

std::vector<std::vector<InvertedBlock>> IndexManager::buildInvertedIndex(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const std::vector<int>& assignments,
    int num_clusters,
    const struct ExecConfig& config
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    // The partitioned index: Concept -> List of Blocks
    std::vector<std::vector<InvertedBlock>> index(n_dims);

    // Strategy change: collect all candidate (doc,weight) per (concept,cluster)
    // then select the best clusters (blocks) per concept and best documents per block
    // according to weight sums / individual weights. This ensures we keep the
    // best `max_blocks_per_dimension` blocks and the best `max_docs_per_block` docs.

    std::cout << "[INDEXING] Collecting candidates for best-block selection..." << std::endl;

    // candidates[concept][cluster] -> vector of (doc_id, weight)
    std::vector<std::unordered_map<int, std::vector<std::pair<int, float>>>> candidates(n_dims);

    for (int i = 0; i < n_docs; ++i) {
        int cluster_id = assignments[i];
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
            int concept_id = it.index();
            float weight = it.value();
            candidates[concept_id][cluster_id].emplace_back(i, weight);
        }
    }

    // For each concept, choose best clusters (by sum of weights) up to limit,
    // and within each cluster keep top documents by individual weight.
    for (int concept_id = 0; concept_id < n_dims; ++concept_id) {
        auto &cluster_map = candidates[concept_id];
        if (cluster_map.empty()) continue;

        // Build vector of (cluster_id, score=sum(weights)) for ranking
        std::vector<std::pair<int, float>> cluster_scores;
        cluster_scores.reserve(cluster_map.size());
        for (auto &p : cluster_map) {
            float sumw = 0.0f;
            for (auto &dw : p.second) sumw += dw.second;
            cluster_scores.emplace_back(p.first, sumw);
        }

        // Determine how many clusters to keep
        int keep_clusters = static_cast<int>(cluster_scores.size());
        if (config.max_blocks_per_dimension > 0 && keep_clusters > config.max_blocks_per_dimension)
            keep_clusters = config.max_blocks_per_dimension;

        // Partially sort to get top-k clusters by sum weight (descending)
        if (keep_clusters < static_cast<int>(cluster_scores.size())) {
            std::nth_element(cluster_scores.begin(), cluster_scores.begin() + keep_clusters, cluster_scores.end(),
                [](const auto &a, const auto &b){ return a.second > b.second; });
            cluster_scores.resize(keep_clusters);
        }

        // Sort the selected clusters by score descending for stable ordering
        std::sort(cluster_scores.begin(), cluster_scores.end(), [](const auto &a, const auto &b){ return a.second > b.second; });

        // Create blocks for the selected clusters
        for (auto &cs : cluster_scores) {
            int cluster_id = cs.first;
            auto &docs = cluster_map[cluster_id];

            // Sort docs by weight descending
            std::sort(docs.begin(), docs.end(), [](const auto &a, const auto &b){ return a.second > b.second; });

            // Determine how many docs to keep
            int keep_docs = static_cast<int>(docs.size());
            if (config.max_docs_per_block > 0 && keep_docs > config.max_docs_per_block)
                keep_docs = config.max_docs_per_block;

            InvertedBlock block;
            block.cluster_id = cluster_id;
            block.doc_ids.reserve(keep_docs);
            block.weights.reserve(keep_docs);

            for (int i = 0; i < keep_docs; ++i) {
                block.doc_ids.push_back(docs[i].first);
                block.weights.push_back(docs[i].second);
            }

            index[concept_id].push_back(std::move(block));
        }
    }

    return index;
}