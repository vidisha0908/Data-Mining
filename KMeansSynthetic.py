# Name: Vidisha
# Student ID: 201709173
# Question 2

# importing the required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from Common_Compiled import KMeans, np_lib

# Set a fixed seed for reproducibility
np_lib.random.seed(24)
k_point= 3
maxIter = 100

# As dataset is big, assuming the n_samples and n_features as 300 and 3, respectively.
n_samples = 300
n_features = 3

# Generating the synthetic dataset
synthetic_data, make_blobs = make_blobs(n_samples=n_samples, n_features=n_features, centers=k_point, random_state= 24)

# Silhouette coefficients lists storing
coefficients_of_silhouette = []

# Silhouette scores calculation for k from 2 to 10
for k_point in range(2, 10):
    cluster_ids, centroids = KMeans(synthetic_data, k_point, maxIter)
    score = silhouette_score(synthetic_data, cluster_ids)
    coefficients_of_silhouette.append(score)

# Plotting the silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 10), coefficients_of_silhouette, marker='o')
plt.xticks(range(2, 10))
plt.xlabel('Number of clusters')
plt.ylabel('Coefficient of Silhouette')
plt.title('Coefficients of Silhouette vs Numbers of Clusters')
plt.savefig('silhouette_scores Q2.png')
plt.show()