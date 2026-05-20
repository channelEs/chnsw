#include "clustering_engine.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h> // For parallelizing on 8 vCPUs

ClusterResult ClusteringEngine::run(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data, 
    int num_clusters, 
    ExecutionProfiler& profiler,
    int max_iterations
) {
    int n_docs = data.rows();
    int n_dims = data.cols();
    
    ClusterResult result;
    result.assignments.resize(n_docs);
    result.centroids.resize(num_clusters, Eigen::VectorXf::Zero(n_dims));

    // 1. Initialization: Pick random documents as initial centroids
    std::cout << "Initializing " << num_clusters << " clusters... with " << n_docs << " documents with " << n_dims << " dimensions" << std::endl;
    std::vector<int> indices(n_docs);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < num_clusters; ++i) {
        result.centroids[i] = Eigen::VectorXf(data.row(indices[i]));
        result.centroids[i].normalize();
    }

    // 2. Iterative Clustering
    for (int iter = 0; iter < max_iterations; ++iter) {
        profiler.start("clustering_iteration_" + std::to_string(iter + 1));
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << max_iterations << " ---" << std::endl;
        
        // =====================================================================
        // Step A: Assignment (Parallelized across 8 Cores)
        // =====================================================================
        std::cout << "Assigning " << n_docs << " documents to clusters (8 threads)..." << std::endl;
        
        // Using dynamic scheduling with a chunk size because document length 
        // (number of non-zero elements) can vary wildly.
        #pragma omp parallel for schedule(dynamic, 128) num_threads(8)
        for (int i = 0; i < n_docs; ++i) {
            float max_sim = -1.0f;
            int best_cluster = 0;
            
            // Eigen handles internal sparse matrix locking for read-only rows natively
            auto doc_row = data.row(i);
            
            for (int j = 0; j < num_clusters; ++j) {
                float sim = doc_row.dot(result.centroids[j]);
                if (sim > max_sim) {
                    max_sim = sim;
                    best_cluster = j;
                }
            }
            result.assignments[i] = best_cluster;
        }
        
        // =====================================================================
        // Step B: Update Centroids (Race-Free Inverted Strategy)
        // =====================================================================
        std::cout << "Grouping documents by cluster assignments..." << std::endl;
        
        // Invert index: Create a quick list of document IDs mapping to each cluster
        std::vector<std::vector<int>> cluster_to_docs(num_clusters);
        for (int i = 0; i < n_docs; ++i) {
            cluster_to_docs[result.assignments[i]].push_back(i);
        }
        
        std::cout << "Updating cluster centroid shapes in parallel..." << std::endl;

        // Process each of the 1,000 clusters entirely independently across threads
        #pragma omp parallel for schedule(dynamic) num_threads(8)
        for (int j = 0; j < num_clusters; ++j) {
            Eigen::VectorXf new_centroid = Eigen::VectorXf::Zero(n_dims);
            const auto& doc_ids = cluster_to_docs[j];
            
            if (!doc_ids.empty()) {
                // Accumulate all sparse document vectors assigned to cluster j
                for (int doc_id : doc_ids) {
                    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, doc_id); it; ++it) {
                        new_centroid[it.index()] += it.value();
                    }
                }
                new_centroid.normalize();
                result.centroids[j] = new_centroid;
            } else {
                // Re-initialize empty cluster safely with a random document selection
                int rand_doc = 0;
                #pragma omp critical(random_gen)
                {
                    rand_doc = rand() % n_docs;
                }
                Eigen::VectorXf random_centroid = Eigen::VectorXf(data.row(rand_doc));
                random_centroid.normalize();
                result.centroids[j] = random_centroid;
            }
        }
        
        std::cout << "Centroids updated successfully!" << std::endl;
        profiler.stop("clustering_iteration_" + std::to_string(iter + 1));
    }
    
    return result;
}