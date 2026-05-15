#include <iostream>
#include "utils/hdf5_sparse_loader.h"
#include "utils/types.h"
#include "clustering_engine.h"
#include "metrics_utils.h"
#include "index_manager.h"

int main() {
    try {
        ExecutionProfiler profiler;
        profiler.start("total_execution");
        std::cout << "loading data..." << std::endl;
        // HDF5SparseLoader loader("data/nq.h5");
        profiler.start("loading_data");
        HDF5SparseLoader loader("data/fiqa-dev.h5");

        auto train = loader.load<float>("train");
        auto query = loader.load<float>("otest/queries");

        std::cout << "train: " << train.rows() << " x " << train.cols() << "\n";
        std::cout << "query: " << query.rows() << " x " << query.cols() << "\n";
        profiler.stop("loading_data");

        // 2. Run Clustering
        int k_clusters = 128; // Start small for testing
        int iterations = 5;
        
        ClusteringEngine engine;
        profiler.start("clustering");
        auto result = engine.run(train, k_clusters, iterations);
        profiler.stop("clustering");

        // 3. Simple Verification
        std::vector<int> counts(k_clusters, 0);
        for (int a : result.assignments) counts[a]++;
        
        std::cout << "\nClustering complete. Sample cluster sizes:\n";
        for (int i = 0; i < k_clusters; ++i) {
            std::cout << "Cluster " << i << ": " << counts[i] << " docs\n";
        }

        auto metrics = ClusterEvaluator::evaluate(train, result);
        std::cout << "Avg Similarity: " << metrics.avg_intra_cluster_similarity << std::endl;
        std::cout << "Balance Score: " << metrics.cluster_balance_score << std::endl;

        // 4. Build Inverted Index
        IndexManager index_manager;

        profiler.start("computing_summaries");
        std::vector<Eigen::VectorXf> summaries = index_manager.computeSummaryVectors(train, result.assignments, k_clusters);
        profiler.stop("computing_summaries");

        profiler.start("building_index");
        auto inverted_index = index_manager.buildInvertedIndex(train, result.assignments, k_clusters);
        profiler.stop("building_index");
        
        std::cout << "Total Summary Vectors: " << summaries.size() << std::endl;
        std::cout << "Total Concepts indexed: " << inverted_index.size() << std::endl;

        // Quick check on a sample concept (e.g., Concept ID 500)
        if (!inverted_index[500].empty()) {
            std::cout << "Sample Concept [500] is present in " << inverted_index[500].size() << " different clusters/blocks.\n";
        }

        profiler.stop("total_execution");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}