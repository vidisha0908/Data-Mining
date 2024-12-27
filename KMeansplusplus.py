# Name: Vidisha
# Student ID: 201709173
# Question 3

# importing the required libraries

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Common_Compiled import np_lib

# Set the random seed to get the same output
np_lib.random.seed(24)

# Assuming data and to perform the KMean Plus
data = np_lib.random.rand(300, 2) * 2 - 1
cluster_ids = np_lib.random.choice([0, 1, 2], 300)
centroids = np_lib.array([[np_lib.random.rand(), np_lib.random.rand()] for _ in range(3)])
def kmeans_plus_plus(data, k, maxIter=100):
    n_samples, data_shape = data.shape

    # Centroids initialization and store the array in list
    centroids = []
    # Selection of the centroids
    centroids.append(data[np_lib.random.randint(0, n_samples)])

    for data_shape in range(1, k):
        distances = np_lib.array([min(np_lib.linalg.norm(x - c) ** 2 for c in centroids) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np_lib.cumsum(probabilities)
        r = np_lib.random.rand()

        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break

    centroids = np_lib.array(centroids)

    # Redefining the clusters
    for data_shape in range(maxIter):
        cluster_ids = np_lib.array([np_lib.argmin([np_lib.linalg.norm(x - centroid) ** 2 for centroid in centroids]) for x in data])
        new_centroids = np_lib.array([data[cluster_ids == i].mean(axis=0) for i in range(k)])
        if np_lib.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return cluster_ids, centroids

# Silhouette scores storing in the list
silhouette_scores = []

# considering range of the k from 1 to 9
for k in range(1, 10):
    if k == 1:
        silhouette_scores.append(-1)  # Silhouette score is not defined for k=1
    else:
        cluster_ids, _ = kmeans_plus_plus(data, k)
        score = silhouette_score(data, cluster_ids)
        silhouette_scores.append(score)

#plotting the graph between Silhouette Coefficient and Number of Clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), silhouette_scores, 'o-')
plt.title('Silhouette Coefficient vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.xticks(range(1, 10))
plt.savefig('silhouette_scores Q3.png')
plt.show()