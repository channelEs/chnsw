#include <iostream>
#include "utils/hdf5_sparse_loader.h"
#include "utils/types.h"
#include "utils/metrics.h"
#include "clustering_engine.h"
#include "index_manager.h"

#include "search_engine.h"
#include "search_engine_simple.h"
#include "search_engine_optimized.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <Eigen/Sparse>

int main() {
    try {
        ExecutionProfiler profiler;
        profiler.start("total_execution");
        // std::cout << "loading data..." << std::endl;
        profiler.start("loading_data");
        std::cout << "\n--- Read data (data/nq.h5) ---" << std::endl;
        HDF5SparseLoader loader("data/nq.h5");
        // std::cout << "\n--- Read data (data/fiqa-dev.h5) ---" << std::endl;
        // HDF5SparseLoader loader("data/fiqa-dev.h5");
        
        auto train = loader.load<float>("train");
        auto query = loader.load<float>("otest/queries");
        
        std::cout << "train: " << train.rows() << " x " << train.cols() << "\n";
        std::cout << "query: " << query.rows() << " x " << query.cols() << "\n";
        profiler.stop("loading_data");
        
        int k_clusters = 1000; // Start small for testing
        int iterations = 10;
        
        std::cout << "\n--- Clustering | k = " << k_clusters << " | iterations = " << iterations << " ---" << std::endl;
        ClusteringEngine engine;
        profiler.start("clustering");
        auto result = engine.run(train, k_clusters,  profiler, iterations);
        profiler.stop("clustering");
        
        std::vector<int> counts(k_clusters, 0);
        for (int a : result.assignments) counts[a]++;
        
        std::cout << "\nClustering complete. Sample cluster sizes:\n";
        for (int i = 0; i < k_clusters; ++i) {
            std::cout << "Cluster " << i << ": " << counts[i] << " docs\n";
        }
        
        std::cout << "\n--- Evaluate clustering ---" << std::endl;
        auto metrics = ClusterEvaluator::evaluate(train, result);
        std::cout << "Avg Similarity: " << metrics.avg_intra_cluster_similarity << std::endl;
        std::cout << "Balance Score: " << metrics.cluster_balance_score << std::endl;
        
        IndexManager index_manager;
        
        std::cout << "\n--- Summary Vectors for each cluster & Indexing ---" << std::endl;
        profiler.start("computing_summaries");
        std::vector<Eigen::VectorXf> summaries = index_manager.computeSummaryVectors(train, result.assignments, k_clusters);
        profiler.stop("computing_summaries");
        
        profiler.start("building_index");
        auto inverted_index = index_manager.buildInvertedIndex(train, result.assignments, k_clusters);
        profiler.stop("building_index");
        
        std::cout << "Total Summary Vectors: " << summaries.size() << std::endl;
        std::cout << "Total Concepts indexed: " << inverted_index.size() << std::endl;

        // ---------- RANDOM QUERY EVALUATION
        int num_eval_queries = query.rows(); // Evaluate on all queries for now
        // if (num_eval_queries > query.rows()) num_eval_queries = query.rows();
        // std::cout << "\n--- Query search and evaluation!! (" << num_eval_queries << " random queries) ---" << std::endl;
        // std::vector<int> random_query_indices(query.rows());
        // std::iota(random_query_indices.begin(), random_query_indices.end(), 0);
        // std::mt19937 g(std::random_device{}());
        // std::shuffle(random_query_indices.begin(), random_query_indices.end(), g);
        // random_query_indices.resize(num_eval_queries);

        // SearchEngineSimple search_engine = SearchEngineSimple();
        SearchEngineOptimized search_engine;

        SearchEngine& search_engine_ref = search_engine; // Polymorphic reference for easy switching between implementations
        float pruning_aggressiveness = 1.0f; // 1.0f preserves exact ceiling limits
        std::vector<std::vector<std::pair<float, int>>> evaluation_results(num_eval_queries);
        
        profiler.start("load_query_gold_standard");
        auto gold_standard = loader.loadGoldStandard("otest/knns");
        profiler.stop("load_query_gold_standard");
        
        profiler.start("search_phase");
        
        // int top_n_to_check = 5; 
        std::vector<int> top_n_values = {10, 30, 100};
        for (int top_n_to_check : top_n_values) {
            std::cout << "\n--- Evaluating Recall@" << top_n_to_check << " ---\n";
            float total_recall = 0.0f;
            profiler.start("search_phase_top_n_" + std::to_string(top_n_to_check));
            for (int actual_query_idx = 0; actual_query_idx < num_eval_queries; ++actual_query_idx) {
                evaluation_results[actual_query_idx] =  search_engine_ref.search(train, inverted_index, summaries, query, actual_query_idx, top_n_to_check, pruning_aggressiveness);
                const auto& hits = evaluation_results[actual_query_idx];
                
                // std::cout << "--- Top " << top_n_to_check << " Evaluation Evaluation for Query " << actual_query_idx << " ---\n";
                auto gold_start = gold_standard[actual_query_idx].begin();
                auto gold_end = gold_start + std::min<size_t>(top_n_to_check, gold_standard[actual_query_idx].size());
                // ----- print the top_n_to_check gold standard doc ids for this query
                // std::cout << "Gold Standard Top-" << top_n_to_check << " DocIDs for Query " << actual_query_idx << ": ";
                // for (auto it = gold_start; it != gold_end; ++it) {
                //     std::cout << *it << " ";            
                // }
                // std::cout << "\n";
                int true_positives = 0;
                int elements_to_check = std::min<int>(top_n_to_check, hits.size());
                
                for (int k = 0; k < elements_to_check; ++k) {
                    int predicted_doc = hits[k].second;
                    bool is_true_positive = (std::find(gold_start, gold_end, predicted_doc) != gold_end);
                    if (is_true_positive) {
                        true_positives++;
                    }
                    // Print the predicted doc id rank position and whether it was found in the top N standard
                    // std::cout << "  Hit Rank " << k + 1 << " -> Predicted DocID: " << predicted_doc << " | " 
                    // << (is_true_positive ? "True Positive (In Top-" + std::to_string(top_n_to_check) + " Gold)" 
                    // : "False Positive") << "\n";
                }
                float query_recall = (top_n_to_check > 0) ? static_cast<float>(true_positives) / static_cast<float>(top_n_to_check) : 0.0f;
                total_recall += query_recall;
                if ((actual_query_idx) % 1000 == 0) {
                    std::cout << "  Processed " << (actual_query_idx) << "/" << num_eval_queries << " queries...\n";
                }
                // std::cout << "  Query " << actual_query_idx << " Recall@" << top_n_to_check << ": " << query_recall << "\n";
            }
                
            profiler.stop("search_phase_top_n_" + std::to_string(top_n_to_check));
            float average_recall = total_recall / static_cast<float>(num_eval_queries);
            
            std::cout << "====================================================\n";
            std::cout << "  VAL RESULTS (N = " << num_eval_queries << " queries)\n";
            std::cout << "  Average Recall@" << top_n_to_check << " = " << average_recall << "\n";
            if (average_recall >= 0.90f) {
                std::cout << "  STATUS: SUCCESS (Passed Challenge Benchmark Threshold)\n";
            } else {
                std::cout << "  STATUS: FAIL (Tune pruning hyperparameter/clustering balance)\n";
            }
            std::cout << "====================================================\n";
        }
        profiler.stop("search_phase");

        profiler.stop("total_execution");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}