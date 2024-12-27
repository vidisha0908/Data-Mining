# Name: Vidisha
# Student ID: 201709173
# Question 1
# importing required libraries for plotting the graphs and data manipulation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import silhouette_score

# importing the required function from the Common_Functions file
from Common_Compiled import KMeans, np_lib, load_dataset

# Using KMeans function to perform the clusters
# number of clusters
k_point = 3
# maximum number of interactions
maxIter = 100

# Set a fixed seed to get same outcomes everytime
np_lib.random.seed(24)

# Assuming data and to perform the clusters and silhouette_score in 2D space
data = np_lib.random.rand(300, 2) * 2 - 1
cluster_ids = np_lib.random.choice([0, 1, 2], 300)
centroids = np_lib.array([[np_lib.random.rand(), np_lib.random.rand()] for _ in range(3)])

cluster_ids, centroids = KMeans(data, k_point, maxIter)

# plotting the three different cluster graphs
def clusters_graph(data, cluster_ids, centroids, colors=None):
    if colors is None:
        colors = ['yellow', 'orange', 'red']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ax in enumerate(axs):
        # Filtering the data points for the current cluster if there will be no data points then it will continue to the next one
        group_data = data[cluster_ids == idx]
        if group_data.size == 0:
            continue

        # Data points plotting
        ax.scatter(group_data[:, 0], group_data[:, 1], s=50, c=colors[idx], edgecolors='brown')
        centroid = centroids[idx]
        circle = Circle(centroid, 0.5, color='black', fill=False, linestyle='--', linewidth=2)
        ax.add_artist(circle)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('on')

    plt.tight_layout()
    plt.suptitle('Outcome of Clustering Algorithm')
    plt.savefig('Outcome of Clustering Algorithm Q1.png')
    plt.show()
def compute_silhouette(data, cluster_ids):
    if len(np_lib.unique(cluster_ids)) > 1:
        return silhouette_score(data, cluster_ids)
    else:
        return -1

def silhouette_scores_graph(data, max_k=9):
    silhouette_scores = []
    for k in range(1, max_k + 1):
        _, centroids = KMeans(data, k)
        cluster_ids, _ = KMeans(data, k)
        score = compute_silhouette(data, cluster_ids)
        silhouette_scores.append(score)

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Various Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_scores Q1.png')
    plt.show()

# Showing the output of the graphs
clusters_graph(data, cluster_ids, centroids)
silhouette_scores_graph(data)