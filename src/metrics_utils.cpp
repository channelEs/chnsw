#include "utils/metrics.h"
#include <algorithm>

ClusterEvaluator::Metrics ClusterEvaluator::evaluate(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const ClusterResult& result
) {
    int n_docs = data.rows();
    int n_clusters = result.centroids.size();
    
    Metrics m;
    m.sizes.assign(n_clusters, 0);
    m.empty_clusters = 0;
    double total_sim = 0;

    // 1. Calculate Sizes and Total Similarity
    for (int i = 0; i < n_docs; ++i) {
        int c = result.assignments[i];
        m.sizes[c]++;
        
        // Dot product between doc and its assigned centroid
        total_sim += data.row(i).dot(result.centroids[c]);
    }

    m.avg_intra_cluster_similarity = total_sim / n_docs;

    // 2. Calculate Balance (Standard Deviation of sizes)
    double avg_size = static_cast<double>(n_docs) / n_clusters;
    double variance = 0;
    for (int size : m.sizes) {
        if (size == 0) m.empty_clusters++;
        variance += std::pow(size - avg_size, 2);
    }
    m.cluster_balance_score = std::sqrt(variance / n_clusters);

    return m;
}