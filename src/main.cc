#include <iostream>
#include "utils/hdf5_sparse_loader.h"
#include "utils/types.h"
#include "clustering_engine.h"
#include "metrics_utils.h"
#include "index_manager.h"
#include "search_engine.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <Eigen/Sparse>

int main() {
    try {
        ExecutionProfiler profiler;
        profiler.start("total_execution");
        std::cout << "\n--- Read data (data/fiqa-dev.h5) ---" << std::endl;
        std::cout << "loading data..." << std::endl;
        // HDF5SparseLoader loader("data/nq.h5");
        profiler.start("loading_data");
        HDF5SparseLoader loader("data/fiqa-dev.h5");
        
        auto train = loader.load<float>("train");
        auto query = loader.load<float>("otest/queries");
        
        std::cout << "train: " << train.rows() << " x " << train.cols() << "\n";
        std::cout << "query: " << query.rows() << " x " << query.cols() << "\n";
        profiler.stop("loading_data");
        
        int k_clusters = 128; // Start small for testing
        int iterations = 5;
        
        std::cout << "\n--- Clustering | k = " << k_clusters << " | iterations = " << iterations << " ---" << std::endl;
        ClusteringEngine engine;
        profiler.start("clustering");
        auto result = engine.run(train, k_clusters, iterations);
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
        
        std::cout << "\nSample Summary Vectors:\n";
        for (int i = 0; i < k_clusters; ++i) {
            // print the number of non zero elements in the summary of cluster i
            std::cout << "Cluster " << i << ": " << summaries[i].count() << " non-zero elements\n";
        }
        // print 10 random concepts from the inverted index
        std::cout << "\nSample Concepts size in Inverted Index:\n";
        for (int i = 0; i < 10; ++i) {
            int idx = rand() % inverted_index.size();
            std::cout << "Sample Concept ["<<idx<<"] is present in " << inverted_index[idx].size() << " different clusters/blocks.\n";
        }

        int num_eval_queries = 20;
        if (num_eval_queries > query.rows()) num_eval_queries = query.rows();
        std::cout << "\n--- Query search and evaluation!! (" << num_eval_queries << " random queries) ---" << std::endl;

        std::vector<int> random_query_indices(query.rows());
        std::iota(random_query_indices.begin(), random_query_indices.end(), 0);
        
        std::mt19937 g(std::random_device{}());
        std::shuffle(random_query_indices.begin(), random_query_indices.end(), g);
        random_query_indices.resize(num_eval_queries);

        SearchEngine search_engine;
        int top_k_demand = 30;
        float pruning_aggressiveness = 1.0f; // 1.0f preserves exact ceiling limits
        std::vector<std::vector<std::pair<float, int>>> evaluation_results(num_eval_queries);

        profiler.start("load_query_gold_standard");
        auto gold_standard = loader.loadGoldStandard("otest/knns");
        profiler.stop("load_query_gold_standard");
        
        profiler.start("search_phase");
        float total_recall = 0.0f;
        
        int top_n_to_check = 5; 
        for (int i = 0; i < num_eval_queries; ++i) {
            int actual_query_idx = random_query_indices[i];
            evaluation_results[i] = search_engine.search(train, inverted_index, summaries, query, actual_query_idx, top_k_demand, pruning_aggressiveness);
            
            // Verification snippet for the first sampled item
            std::cout << "\n[VERIFICATION] Top 5 matches for Sampled Query Index [" << actual_query_idx << "]:\n";
            for (size_t r = 0; r < std::min<size_t>(5, evaluation_results[i].size()); ++r) {
                std::cout << "  Rank " << r + 1 << ": DocID = " << evaluation_results[i][r].second 
                          << " | Similarity Score = " << evaluation_results[i][r].first << "\n";
            }
            std::cout << std::endl;

            const auto& hits = evaluation_results[i];
            
            std::cout << "--- Top " << top_n_to_check << " Evaluation Evaluation for Query " << actual_query_idx << " ---\n";
            auto gold_start = gold_standard[actual_query_idx].begin();
            auto gold_end = gold_start + std::min<size_t>(top_n_to_check, gold_standard[actual_query_idx].size());
            
            int true_positives = 0;
            int elements_to_check = std::min<int>(top_n_to_check, hits.size());

            for (int k = 0; k < elements_to_check; ++k) {
                int predicted_doc = hits[k].second;
                bool is_true_positive = (std::find(gold_start, gold_end, predicted_doc) != gold_end);
                if (is_true_positive) {
                    true_positives++;
                }
                // Print the predicted doc id rank position and whether it was found in the top N standard
                std::cout << "  Hit Rank " << k + 1 << " -> Predicted DocID: " << predicted_doc << " | " 
                          << (is_true_positive ? "True Positive (In Top-" + std::to_string(top_n_to_check) + " Gold)" 
                                               : "False Positive") << "\n";
            }
            float query_recall = (top_n_to_check > 0) ? static_cast<float>(true_positives) / static_cast<float>(top_n_to_check) : 0.0f;
            total_recall += query_recall;
            std::cout << "  Query Recall@" << top_n_to_check << ": " << query_recall << "\n\n";
        }
        profiler.stop("search_phase");

        float average_recall = total_recall / static_cast<float>(num_eval_queries);

        std::cout << "====================================================\n";
        std::cout << "  VAL RESULTS (N = " << num_eval_queries << " random queries)\n";
        std::cout << "  Average Recall@" << top_k_demand << " = " << average_recall << "\n";
        if (average_recall >= 0.90f) {
            std::cout << "  STATUS: SUCCESS (Passed Challenge Benchmark Threshold)\n";
        } else {
            std::cout << "  STATUS: FAIL (Tune pruning hyperparameter/clustering balance)\n";
        }
        std::cout << "====================================================\n";

        profiler.stop("total_execution");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}