#include "clustering_engine.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h> // For parallelizing on 8 vCPUs

ClusteringEngine::ClusterResult ClusteringEngine::run(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data, 
    int num_clusters, 
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
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (int i = 0; i < num_clusters; ++i) {
        result.centroids[i] = Eigen::VectorXf(data.row(indices[i]));
        result.centroids[i].normalize();
    }

    // 2. Iterative Clustering
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "Iteration " << (iter + 1) << "/" << max_iterations << "..." << std::endl;

        // Step A: Assignment (Parallelized)
        // #pragma omp parallel for schedule(dynamic, 1024)
        std::cout << "Assigning documents to clusters..." << std::endl;
        for (int i = 0; i < n_docs; ++i) {
            float max_sim = -1.0f;
            int best_cluster = 0;
            
            auto doc_row = data.row(i);
            
            for (int j = 0; j < num_clusters; ++j) {
                // Efficient Sparse-Dense dot product
                float sim = doc_row.dot(result.centroids[j]);
                if (sim > max_sim) {
                    max_sim = sim;
                    best_cluster = j;
                }
            }
            result.assignments[i] = best_cluster;
        }
        
        // Step B: Update Centroids
        std::vector<Eigen::VectorXf> new_centroids(num_clusters, Eigen::VectorXf::Zero(n_dims));
        std::vector<int> cluster_sizes(num_clusters, 0);
        
        std::cout << "Updating centroids..." << std::endl;
        for (int i = 0; i < n_docs; ++i) {
            int c = result.assignments[i];
            // Add sparse row to dense centroid
            for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
                new_centroids[c][it.index()] += it.value();
            }
            cluster_sizes[c]++;
        }

        for (int j = 0; j < num_clusters; ++j) {
            if (cluster_sizes[j] > 0) {
                new_centroids[j].normalize();
                result.centroids[j] = new_centroids[j];
            } else {
                // Re-initialize empty cluster with a random doc
                result.centroids[j] = Eigen::VectorXf(data.row(rand() % n_docs));
                result.centroids[j].normalize();
            }
        }

        std::cout << "Centroids updated!" << std::endl;
        // for (int j = 0; j < num_clusters; ++j) {
        //     std::ptrdiff_t nnz_tol = (result.centroids[j].array().abs() > 1e-6f).count();
        //     std::cout << "cluster " << j << " with num of documents: " << cluster_sizes[j] << " with centroid non-zero elements: " << nnz_tol << std::endl;
        // }
        // std::cout << std::endl;
    }

    return result;
}