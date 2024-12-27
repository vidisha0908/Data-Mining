# Name: Vidisha
# Student ID: 201709173

'''This script is used for the common functions and contains the compiled code to run all the questions in one click to get the outputs
for question 1, 2, 3 and 4.'''

# COMMON FUNCTIONS
import numpy as np_lib
import os
'''The functions were pre-defined and same has been taken from the assignment pdf'''
def computeDistance(p, q):
    # calculate the difference between two arrays by considering the arrays to be of same dimension.
    # Taking square root of each element and then summing up the squared difference.
    # Square root of the sum of squares- to converts the squared Euclidean distance back to the original scale.
    return np_lib.sqrt(np_lib.sum((p - q) ** 2))

def initialisation(data, k_point):
    # To avoid selecting the same index more than once and creating duplicate centroids, choose 'k_point' indices at random from the data's length range without replacement.
    indices = np_lib.random.choice(len(data), k_point, replace=False)
    centroids = data[indices]
    return centroids

def computeClusterRepresentatives(data, cluster_ids, k_point):
    # This will calculate the mean of the data points and will return a new centroids.
    # Each centroid is the average position of all data points in that particular cluster.
    centroids = np_lib.array([data[cluster_ids == i].mean(axis=0) for i in range(k_point)])
    return centroids

def assignClusterIds(data, centroids):
    # calculating nearest centroid for each data point in the dataset.
    # For each data point 'X' computing the Euclidean distance to each centroid and then find the index with the minimum distance.
    cluster_ids = np_lib.array([np_lib.argmin([computeDistance(x, centroid) for centroid in centroids]) for x in data])
    return cluster_ids

def KMeans(data, k_point= 3, maxIter=100):
    # Initialize centroids by randomly selecting 'k_point' points and maxInter
    centroids = initialisation(data, k_point)
    for _ in range(maxIter):
        cluster_ids = assignClusterIds(data, centroids)
        centroids = computeClusterRepresentatives(data, cluster_ids, k_point)
    return cluster_ids, centroids


'''Loading the dataset'''
def load_dataset():
    # Accessing the dataset from the current folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'dataset')

    # To store headers and data independently, initialize two empty lists because of the mismatch of data type.
    headers = []
    data_values = []

    # opening the file in read mode from the file path
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Appending the headers
            headers.append(parts[0])
            # Appending as a list to the value list
            data_values.append([float(x) for x in parts[1:]])

    '''converting the headers and value lists to numpy arrays before loading the dataset to use.
    This allows to perform mathematical calculations and data manipulation in the further questions.'''
    return np_lib.array(headers), np_lib.array(data_values)

# loading the data from the local device
headers, data_values = load_dataset()

# COMPILED CODE
''' Run this code to get all outputs for question 1, 2, 3 and 4 in one click '''

import subprocess
import sys

def Compiled_Script(File_name, python_path):
    # Runs KMeans, KMeansSynthetic, KMeansplusplus, BisectingKMeans
    try:
        # Run the script
        result = subprocess.run([python_path, File_name], text=True, capture_output=True, check=True)
        return result.stdout  # Return the output of the script
    except subprocess.CalledProcessError as e:
        # Return the comment mentioned if get errors in the execution of the scripts
        return f"An error occurred while running {File_name}: {e}"

def main():
    # Listed Question 1 to Question 4 file names
    Files = ['KMeans.py', 'KMeansSynthetic.py', 'KMeansplusplus.py', 'BisectingKMeans.py']
    python_path = sys.executable

    # Capture the output in txt file-just in case
    with open('Compiled_outputs.txt', 'w') as file:
        for py_script in Files:
            print(f"Running {py_script}...")
            output = Compiled_Script(py_script, python_path)
            file.write(f"Output of {py_script}:\n{output}\n")
            print(f"Output of {py_script} captured to compiled_outputs.txt.")

if __name__ == "__main__":
    main()
