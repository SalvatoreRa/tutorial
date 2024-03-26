# -*- coding: utf-8 -*-

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np

def normalized_stress(original_data, projected_data):
    """
    Calculate the normalized stress between the original high-dimensional data
    and its low-dimensional projection.

    Parameters:
    - original_data: numpy array of shape (n_samples, n_features)
    - projected_data: numpy array of shape (n_samples, n_components)

    Returns:
    - normalized_stress: float
    """
    
    distances_original = euclidean_distances(original_data)
    distances_projected = euclidean_distances(projected_data)
    
   
    sum_squared_distances_original = np.sum(distances_original ** 2)
    sum_squared_distance_differences = np.sum((distances_original - distances_projected) ** 2)
    
    
    stress = np.sqrt(sum_squared_distance_differences / sum_squared_distances_original)
    
    return stress


def continuity(original_data, projected_data, k=5):
    """
    Compute the Continuity metric to evaluate the quality of a low-dimensional projection.
    
    Parameters:
    - original_data: numpy array of shape (n_samples, n_features)
    - projected_data: numpy array of shape (n_samples, n_components)
    - k: Number of nearest neighbors to consider
    
    Returns:
    - Continuity score
    """
    
    distances_high = pairwise_distances(original_data)
    distances_low = pairwise_distances(projected_data)
    
  
    ranks_high = np.argsort(distances_high, axis=1)
    ranks_low = np.argsort(distances_low, axis=1)
    
    n_samples = original_data.shape[0]
    continuity_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
       
        high_neighbors = ranks_high[i, 1:k+1]
        
        low_ranks = np.array([np.where(ranks_low[i] == neighbor)[0][0] for neighbor in high_neighbors])
        

        continuity_scores[i] = np.mean(low_ranks)
    

    max_rank = n_samples - 1
    normalized_continuity = 1 - np.mean(continuity_scores) / max_rank
    
    return normalized_continuity


def neighborhood_hit(original_data, projected_data, n_neighbors=5):
    """
    Calculate the Neighborhood Hit metric for assessing the quality of a
    low-dimensional projection.

    Parameters:
    - original_data: numpy array of shape (n_samples, n_features)
    - projected_data: numpy array of shape (n_samples, n_components)
    - n_neighbors: Number of nearest neighbors to consider, including the point itself

    Returns:
    - nhit: The Neighborhood Hit score as a float
    """
    nn_high = NearestNeighbors(n_neighbors=n_neighbors).fit(original_data)
    distances_high, indices_high = nn_high.kneighbors(original_data)

    nn_low = NearestNeighbors(n_neighbors=n_neighbors).fit(projected_data)
    distances_low, indices_low = nn_low.kneighbors(projected_data)


    hits = 0
    for i in range(original_data.shape[0]):

        if indices_high[i, 1] in indices_low[i, 1:]:
            hits += 1

    nhit = hits / original_data.shape[0]
    return nhit


def plot_shepard_diagram(original_data, projected_data):
    """
    Plots a Shepard diagram to visualize the relationship between distances in
    the original high-dimensional data space and the projected low-dimensional space.
    
    Parameters:
    - original_data: numpy array of the original high-dimensional data.
    - projected_data: numpy array of the data projected into a lower-dimensional space.
    """

    distances_original = squareform(pdist(original_data, 'euclidean'))
    distances_projected = squareform(pdist(projected_data, 'euclidean'))
    

    distances_original_flat = distances_original.flatten()
    distances_projected_flat = distances_projected.flatten()
    

    plt.figure(figsize=(10, 6))
    plt.scatter(distances_original_flat, distances_projected_flat, alpha=0.5)
    plt.title('Shepard Diagram')
    plt.xlabel('Original Distances')
    plt.ylabel('Projected Distances')
    plt.grid(True)
    plt.show()