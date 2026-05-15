#include "index_manager.h"
#include <iostream>
#include <algorithm>

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
    int num_clusters
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    // The partitioned index: Concept -> List of Blocks
    std::vector<std::vector<InvertedBlock>> index(n_dims);

    // To maintain high performance while building, we need to know 
    // which block in index[concept] corresponds to which cluster.
    // map_block_pos[concept][cluster_id] = index in index[concept]
    std::vector<std::unordered_map<int, int>> map_block_pos(n_dims);

    std::cout << "[INDEXING] Partitioning Inverted Lists into blocks..." << std::endl;

    for (int i = 0; i < n_docs; ++i) {
        int cluster_id = assignments[i];

        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
            int concept_id = it.index();
            float weight = it.value();

            // Check if we already have a block for this cluster in this concept's list
            if (map_block_pos[concept_id].find(cluster_id) == map_block_pos[concept_id].end()) {
                // Create a new block for this cluster
                map_block_pos[concept_id][cluster_id] = index[concept_id].size();
                index[concept_id].push_back({cluster_id, {}, {}});
            }

            // Append document data to the correct block
            int pos = map_block_pos[concept_id][cluster_id];
            index[concept_id][pos].doc_ids.push_back(i);
            index[concept_id][pos].weights.push_back(weight);
        }
    }

    return index;
}