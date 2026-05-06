#include <iostream>
#include "utils/hdf5_sparse_loader.h"
#include "utils/types.h"
#include "clustering_engine.h"

int main() {
    try {
        std::cout << "loading data..." << std::endl;
        // HDF5SparseLoader loader("data/nq.h5");
        HDF5SparseLoader loader("data/fiqa-dev.h5");

        auto train = loader.load<float>("train");
        auto query = loader.load<float>("otest/queries");

        std::cout << "train: " << train.rows() << " x " << train.cols() << "\n";
        std::cout << "query: " << query.rows() << " x " << query.cols() << "\n";

        // 2. Run Clustering
        int k_clusters = 128; // Start small for testing
        int iterations = 5;
        
        ClusteringEngine engine;
        auto result = engine.run(train, k_clusters, iterations);

        // 3. Simple Verification
        std::vector<int> counts(k_clusters, 0);
        for (int a : result.assignments) counts[a]++;
        
        std::cout << "\nClustering complete. Sample cluster sizes:\n";
        for (int i = 0; i < std::min(10, k_clusters); ++i) {
            std::cout << "Cluster " << i << ": " << counts[i] << " docs\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}