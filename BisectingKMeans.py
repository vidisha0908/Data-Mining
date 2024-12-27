# Name: Vidisha
# Student ID: 201709173
# Question 4

# importing the required libraries

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Common_Compiled import KMeans, load_dataset, np_lib

# Set the random seed to get the same output
np_lib.random.seed(24)

# Assuming data and to perform the KMean Plus
data = np_lib.random.rand(300, 2) * 2 - 1
cluster_ids = np_lib.random.choice([0, 1, 2], 300)
centroids = np_lib.array([[np_lib.random.rand(), np_lib.random.rand()] for _ in range(3)])

# calculating bisecting kmeans
def bisecting_kmeans(data, n_clusters):
    # calculating the number of sample in the dataset
    n_samples = data.shape[0]
    # Start with the one cluster including every data points
    clusters = [np_lib.arange(n_samples)]
    # continuing clusters until the desired outcomes reached
    while len(clusters) < n_clusters:
        # largest cluster to split and apply KMean on that
        largest_cluster_idx = np_lib.argmax([len(cluster) for cluster in clusters])
        largest_cluster = clusters.pop(largest_cluster_idx)
        labels, _ = KMeans(data[largest_cluster], 2)

        # add them back to the clusters list based on the division of the cluster's labels
        clusters.append(largest_cluster[labels == 0])
        clusters.append(largest_cluster[labels == 1])
    return clusters

# computing silhouette scores to plot the graph
def compute_silhouette_scores(data, cluster_hierarchy):
    silhouette_scores = []
    n_samples = data.shape[0]

    for num_clusters in range(1, 10):
        # Empty cluster ID array creation
        cluster_ids = np_lib.zeros(n_samples, dtype=int)

        # Assigning the cluster IDs as per the cluster hierarchy
        for idx, cluster in enumerate(cluster_hierarchy[:num_clusters]):
            cluster_ids[cluster] = idx
        if num_clusters == 1:
            silhouette_scores.append(-1)
        else:
            score = silhouette_score(data, cluster_ids)
            silhouette_scores.append(score)

    return silhouette_scores

# Perform bisecting K-means clustering
cluster_hierarchy = bisecting_kmeans(data, 9)

# Calculating the silhouette scores
silhouette_scores = compute_silhouette_scores(data, cluster_hierarchy)

# Plot the silhouette scores graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), silhouette_scores, 'o-')
plt.title('Silhouette Coefficient vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Coefficient of Silhouette')
plt.xticks(range(1, 10))
plt.savefig('silhouette_scores Q4.png')
plt.show()