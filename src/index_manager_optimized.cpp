#include "index_manager_optimized.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <cstdint>

static double estimateIndexMemoryGBytes(int n_dims, int num_clusters, const struct ExecConfig& config) {
    uint64_t max_docs = (config.max_docs_per_block > 0) ? static_cast<uint64_t>(config.max_docs_per_block) : static_cast<uint64_t>(n_dims);
    uint64_t cluster_count = static_cast<uint64_t>(num_clusters);

    unsigned long long pair_size = static_cast<unsigned long long>(sizeof(std::pair<int, float>));
    unsigned long long estimate_docs = static_cast<unsigned long long>(n_dims) * cluster_count * max_docs;
    unsigned long long docs_bytes = estimate_docs * (pair_size * 2); // account for heap/vector overhead

    unsigned long long cluster_nodes = static_cast<unsigned long long>(n_dims) * cluster_count;
    unsigned long long cluster_overhead = cluster_nodes * 80ULL;

    unsigned long long total = docs_bytes + cluster_overhead;
    return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<Eigen::VectorXf> IndexManagerOptimized::computeSummaryVectors(
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

std::vector<std::vector<InvertedBlock>> IndexManagerOptimized::buildInvertedIndex(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const std::vector<int>& assignments,
    int num_clusters,
    const struct ExecConfig& config,
    const std::vector<Eigen::VectorXf>& summaries
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    std::cout << "[INDEXING] buildInvertedIndex start: n_docs=" << n_docs
              << " n_dims=" << n_dims
              << " nonzeros=" << data.nonZeros()
              << " max_blocks_per_dimension=" << config.max_blocks_per_dimension
              << " max_docs_per_block=" << config.max_docs_per_block
              << " max_docs_to_visit=" << config.max_docs_to_visit
              << std::endl;

    const double estimated_index_gbytes = estimateIndexMemoryGBytes(n_dims, num_clusters, config);
    std::cout << "[MEMORY_ESTIMATE] Estimated build memory=" << estimated_index_gbytes
              << " GB. Proceeding with index build regardless of system limits." << std::endl;

    try {
        // The partitioned index: Concept -> List of Blocks
        std::vector<std::vector<InvertedBlock>> index(n_dims);

        // Strategy change: collect all candidate (doc,weight) per (concept,cluster)
        // then select the best clusters (blocks) per concept and best documents per block
        // according to weight sums / individual weights. This ensures we keep the
        // best `max_blocks_per_dimension` blocks and the best `max_docs_per_block` docs.

        std::cout << "[INDEXING] Collecting candidates for best-block selection..." << std::endl;

        struct ClusterCandidate {
            float total_weight = 0.0f;
            std::vector<std::pair<int, float>> docs;
        };

        using DocWeight = std::pair<int, float>;
        auto doc_cmp_min = [](const DocWeight &a, const DocWeight &b) {
            return a.second > b.second;
        };

        std::vector<std::unordered_map<int, ClusterCandidate>> candidates(n_dims);

        const bool has_doc_bound = (config.max_docs_per_block > 0);
        for (int i = 0; i < n_docs; ++i) {
            int cluster_id = assignments[i];
            for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
                int concept_id = it.index();
                float weight = it.value();
                auto &candidate = candidates[concept_id][cluster_id];
                candidate.total_weight += weight;

                if (!has_doc_bound) {
                    candidate.docs.emplace_back(i, weight);
                } else {
                    auto &docs = candidate.docs;
                    if (static_cast<int>(docs.size()) < config.max_docs_per_block) {
                        docs.emplace_back(i, weight);
                        std::push_heap(docs.begin(), docs.end(), doc_cmp_min);
                    } else if (weight > docs.front().second) {
                        std::pop_heap(docs.begin(), docs.end(), doc_cmp_min);
                        docs.back() = {i, weight};
                        std::push_heap(docs.begin(), docs.end(), doc_cmp_min);
                    }
                }
            }
        }

        int non_empty_concepts = 0;
        for (auto &cluster_map : candidates) {
            if (!cluster_map.empty()) {
                ++non_empty_concepts;
            }
        }
        std::cout << "[INDEXING] Candidate collection complete. Non-empty concepts=" << non_empty_concepts << std::endl;
        std::cout << "[INDEXING] Selecting top blocks and documents per concept..." << std::endl;

        int processed_concepts = 0;
        for (int concept_id = 0; concept_id < n_dims; ++concept_id) {
            auto &cluster_map = candidates[concept_id];
            if (cluster_map.empty()) continue;
            ++processed_concepts;
            // if ((processed_concepts <= 5) || (processed_concepts % 1000 == 0)) {
            //     size_t doc_entries = 0;
            //     for (auto &entry : cluster_map) {
            //         doc_entries += entry.second.docs.size();
            //     }
            //     std::cout << "[INDEXING] Processing concept " << concept_id
            //               << " / " << n_dims
            //               << " clusters=" << cluster_map.size()
            //               << " docs=" << doc_entries
            //               << " processed_concepts=" << processed_concepts
            //               << std::endl;
            // }

            std::vector<std::pair<int, ClusterCandidate>> cluster_entries;
            cluster_entries.reserve(cluster_map.size());
            for (auto &p : cluster_map) {
                auto &docs = p.second.docs;
                if (has_doc_bound && !docs.empty()) {
                    std::sort(docs.begin(), docs.end(), [](const DocWeight &a, const DocWeight &b) {
                        return a.second > b.second;
                    });
                }
                cluster_entries.emplace_back(p.first, std::move(p.second));
            }

            int keep_clusters = static_cast<int>(cluster_entries.size());
            if (config.max_blocks_per_dimension > 0 && keep_clusters > config.max_blocks_per_dimension) {
                auto nth = cluster_entries.begin() + config.max_blocks_per_dimension;
                std::nth_element(cluster_entries.begin(), nth, cluster_entries.end(),
                    [](const auto &a, const auto &b) {
                        return a.second.total_weight > b.second.total_weight;
                    });
                cluster_entries.erase(nth, cluster_entries.end());
                keep_clusters = config.max_blocks_per_dimension;
            }

            std::sort(cluster_entries.begin(), cluster_entries.end(), [](const auto &a, const auto &b) {
                return a.second.total_weight > b.second.total_weight;
            });

            for (auto &entry : cluster_entries) {
                int cluster_id = entry.first;
                auto &candidate = entry.second;
                auto &docs = candidate.docs;

                if (!has_doc_bound) {
                    std::sort(docs.begin(), docs.end(), [](const DocWeight &a, const DocWeight &b) {
                        return a.second > b.second;
                    });
                }

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

        // Profiling: compute totals and averages for blocks and documents
        size_t total_blocks = 0;
        size_t total_docs_in_blocks = 0;
        int empty_dimensions = 0;
        for (int concept_id = 0; concept_id < n_dims; ++concept_id) {
            auto &blocks = index[concept_id];
            if (blocks.empty()) {
                ++empty_dimensions;
                continue;
            }
            total_blocks += blocks.size();
            for (const auto &b : blocks) {
                total_docs_in_blocks += b.doc_ids.size();
            }
        }

        double avg_docs_per_block = 0.0;
        if (total_blocks > 0) avg_docs_per_block = static_cast<double>(total_docs_in_blocks) / static_cast<double>(total_blocks);
        double avg_blocks_per_dimension = static_cast<double>(total_blocks) / static_cast<double>(n_dims);

        std::cout << "[INDEX_PROFILE] total_blocks=" << total_blocks
                  << " total_docs_in_blocks=" << total_docs_in_blocks
                  << " avg_docs_per_block=" << avg_docs_per_block
                  << " avg_blocks_per_dimension=" << avg_blocks_per_dimension
                  << " empty_dimensions=" << empty_dimensions << std::endl;

        return index;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] buildInvertedIndex ran out of memory: " << e.what() << std::endl;
        std::cerr << "[ERROR] n_docs=" << n_docs
                  << " n_dims=" << n_dims
                  << " max_blocks_per_dimension=" << config.max_blocks_per_dimension
                  << " max_docs_per_block=" << config.max_docs_per_block
                  << " max_docs_to_visit=" << config.max_docs_to_visit
                  << std::endl;
        throw;
    }
}