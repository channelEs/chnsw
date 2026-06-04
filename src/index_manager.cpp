#include "index_manager.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <cstdint>

static uint64_t parseUnsignedLongLong(const std::string& text) {
    try {
        size_t pos = 0;
        uint64_t value = std::stoull(text, &pos, 10);
        return value;
    } catch (...) {
        return 0;
    }
}

static uint64_t readUnsignedFromFile(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return 0;
    }

    if (line == "max") {
        return 0;
    }

    return parseUnsignedLongLong(line);
}

static uint64_t getCgroupMemoryLimitBytes() {
    uint64_t limit = readUnsignedFromFile("/sys/fs/cgroup/memory.max");
    if (limit > 0) {
        return limit;
    }
    limit = readUnsignedFromFile("/sys/fs/cgroup/memory.limit_in_bytes");
    return limit;
}

static uint64_t getPhysicalMemoryBytes() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
}

static uint64_t getEffectiveMemoryLimitBytes() {
    uint64_t cgroup_limit = getCgroupMemoryLimitBytes();
    if (cgroup_limit > 0 && cgroup_limit < std::numeric_limits<uint64_t>::max()) {
        return cgroup_limit;
    }
    return getPhysicalMemoryBytes();
}

static uint64_t estimateIndexMemoryBytes(int n_dims, int num_clusters, const struct ExecConfig& config) {
    uint64_t max_docs = (config.max_docs_per_block > 0) ? static_cast<uint64_t>(config.max_docs_per_block) : static_cast<uint64_t>(n_dims);
    uint64_t cluster_count = static_cast<uint64_t>(num_clusters);

    unsigned long long pair_size = static_cast<unsigned long long>(sizeof(std::pair<int, float>));
    unsigned long long estimate_docs = static_cast<unsigned long long>(n_dims) * cluster_count * max_docs;
    unsigned long long docs_bytes = estimate_docs * (pair_size * 2); // account for heap/vector overhead

    unsigned long long cluster_nodes = static_cast<unsigned long long>(n_dims) * cluster_count;
    unsigned long long cluster_overhead = cluster_nodes * 80ULL;

    unsigned long long total = docs_bytes + cluster_overhead;
    return total;
}

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

    std::cout << "[INDEXING] buildInvertedIndex start: n_docs=" << n_docs
              << " n_dims=" << n_dims
              << " nonzeros=" << data.nonZeros()
              << " max_blocks_per_dimension=" << config.max_blocks_per_dimension
              << " max_docs_per_block=" << config.max_docs_per_block
              << " max_docs_to_visit=" << config.max_docs_to_visit
              << std::endl;

    const uint64_t memory_limit_bytes = getEffectiveMemoryLimitBytes();
    const uint64_t estimated_index_bytes = estimateIndexMemoryBytes(n_dims, num_clusters, config);
    uint64_t safe_threshold = 0;
    if (memory_limit_bytes > 0) {
        safe_threshold = memory_limit_bytes * 80ULL / 100ULL;
        std::cout << "[MEMORY_ESTIMATE] Estimated build memory=" << estimated_index_bytes
                  << " bytes, limit=" << memory_limit_bytes
                  << " bytes, safe threshold=" << safe_threshold << " bytes" << std::endl;
        if (estimated_index_bytes > safe_threshold) {
            throw std::runtime_error("Estimated index build memory exceeds safe threshold.");
        }
    } else {
        safe_threshold = 128ULL * 1024ULL * 1024ULL * 1024ULL;
        std::cout << "[MEMORY_ESTIMATE] Estimated build memory=" << estimated_index_bytes
                  << " bytes, no cgroup limit available, using safety threshold=" << safe_threshold << " bytes" << std::endl;
        if (estimated_index_bytes > safe_threshold) {
            throw std::runtime_error("Estimated index build memory exceeds local safety limit.");
        }
    }

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
            if ((processed_concepts <= 5) || (processed_concepts % 1000 == 0)) {
                size_t doc_entries = 0;
                for (auto &entry : cluster_map) {
                    doc_entries += entry.second.docs.size();
                }
                std::cout << "[INDEXING] Processing concept " << concept_id
                          << " / " << n_dims
                          << " clusters=" << cluster_map.size()
                          << " docs=" << doc_entries
                          << " processed_concepts=" << processed_concepts
                          << std::endl;
            }

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