#################################################
##### K-means clustering
#####
#################################################



kmeans <- function(k, max.iter = 100, tol = 1e-6) {
  # This class is designed to perform k-means clustering. 
  # An object of this class is created using the constructor function kmeans, 
  # and the clustering operation is performed using the fit method.
  # Parameters:
  # k (integer): Number of clusters.
  # max.iter (integer, default = 100): The maximum number of iterations the algorithm should run. 
  # tol (numeric, default = 1e-6): The tolerance for convergence. If the change in centroids between iterations is less than this value, 
  #     the algorithm is considered to have converged and stops running.
  # example usage.
  # 
  # Create a kmeans object
  # km <- kmeans(k = 3)
  # Generate some data
  # set.seed(123)
  # X <- matrix(rnorm(100), nrow = 10)
  # Fit the model
  # result <- fit(km, X)
  
  # Print results
  # print(result$cluster)  # Cluster assignment for each example
  # print(result$centers)  # Final centroids of the clusters
  
  if (k <= 0) stop("Number of clusters 'k' must be positive.")
  if (max.iter <= 0) stop("Maximum iterations 'max.iter' must be positive.")
  if (tol <= 0) stop("Tolerance 'tol' must be positive.")
  
  # Create an object with the specified properties
  structure(list(k = k, max.iter = max.iter, tol = tol), class = "kmeans")
}

# Define the 'fit' method for the 'kmeans' class
fit <- function(object, X) {
  if (!inherits(object, "kmeans")) stop("Object must be of class 'kmeans'.")
  
  # Extract k, max.iter, and tol from the object
  k <- object$k
  max.iter <- object$max.iter
  tol <- object$tol
  
  # Initialize centroids randomly
  n <- nrow(X)
  centroids <- X[sample(n, k), ]
  old_centroids <- matrix(0, nrow = k, ncol = ncol(X))
  
  # Initialize labels and iteration counter
  labels <- rep(0, n)
  iter <- 0
  converged <- FALSE
  
  # Main k-means iteration loop
  while (!converged && iter < max.iter) {
    # Assign labels based on closest centroid
    for (i in 1:n) {
      dists <- apply(centroids, 1, function(centroid) sum((X[i, ] - centroid)^2))
      labels[i] <- which.min(dists)
    }
    
    # Update centroids
    old_centroids <- centroids
    for (j in 1:k) {
      centroids[j, ] <- colMeans(X[labels == j, ], na.rm = TRUE)
    }
    
    # Check for convergence
    converged <- all(sqrt(rowSums((centroids - old_centroids)^2)) < tol)
    iter <- iter + 1
  }
  
  # Store the result in the object
  object$centers <- centroids
  object$cluster <- labels
  object
}

#################################################
##### bisecting K-means clustering
#####
#################################################

bisecting_kmeans <- function(k, max.iter = 100, tol = 1e-6) {
  # This class is designed to perform bisecting k-means clustering, a variant of the k-means
  # An object of this class is created using the constructor function kmeans, 
  # and the clustering operation is performed using the fit method.
  # Parameters:
  # k (integer): Number of clusters.
  # max.iter (integer, default = 100): The maximum number of iterations the algorithm should run. 
  # tol (numeric, default = 1e-6): The tolerance for convergence. If the change in centroids between iterations is less than this value, 
  #     the algorithm is considered to have converged and stops running.
  # example usage.
  # 
  # Create a kmeans object
  # km <- bisecting_kmeans(k = 3)
  # Generate some data
  # set.seed(123)
  # X <- matrix(rnorm(100), nrow = 10)
  # Fit the model
  # result <- fit(km, X)
  
  if (k <= 0) stop("Number of clusters 'k' must be positive.")
  if (max.iter <= 0) stop("Maximum iterations 'max.iter' must be positive.")
  if (tol <= 0) stop("Tolerance 'tol' must be positive.")
  
  structure(list(k = k, max.iter = max.iter, tol = tol, centers = NULL, cluster = NULL), class = "bisecting_kmeans")
}

# Define the 'fit' method for the 'bisecting_kmeans' class
fit <- function(object, X) {
  if (!inherits(object, "bisecting_kmeans")) stop("Object must be of class 'bisecting_kmeans'.")
  
  # Internal function to perform k-means on a single cluster
  single_kmeans <- function(data, max.iter, tol) {
    n <- nrow(data)
    k <- 2
    centroids <- data[sample(n, k), ]
    old_centroids <- matrix(0, nrow = k, ncol = ncol(data))
    labels <- rep(0, n)
    iter <- 0
    converged <- FALSE
    
    while (!converged && iter < max.iter) {
      for (i in 1:n) {
        dists <- apply(centroids, 1, function(centroid) sum((data[i, ] - centroid)^2))
        labels[i] <- which.min(dists)
      }
      
      old_centroids <- centroids
      for (j in 1:k) {
        centroids[j, ] <- colMeans(data[labels == j, ], na.rm = TRUE)
      }
      
      converged <- all(sqrt(rowSums((centroids - old_centroids)^2)) < tol)
      iter <- iter + 1
    }
    
    list(centers = centroids, cluster = labels)
  }
  
  # Bisecting step
  cluster_list <- list(X)
  while (length(cluster_list) < object$k) {
    largest_cluster_index <- which.max(sapply(cluster_list, nrow))
    largest_cluster <- cluster_list[[largest_cluster_index]]
    cluster_list <- cluster_list[-largest_cluster_index]
    
    # Apply single k-means to the largest cluster
    kmeans_result <- single_kmeans(largest_cluster, object$max.iter, object$tol)
    for (i in 1:2) {
      cluster_list <- c(cluster_list, list(largest_cluster[kmeans_result$cluster == i, , drop = FALSE]))
    }
  }
  
  # Final assignments and centers
  object$centers <- lapply(cluster_list, function(cluster) colMeans(cluster))
  object$cluster <- rep(0, nrow(X))
  idx <- 1
  for (i in 1:length(cluster_list)) {
    object$cluster[which(X == cluster_list[[i]], arr.ind = TRUE)[, 1]] <- i
  }
  
  object
}

#################################################
##### Agglomerative clustering
#####
#################################################

agglo_clustering <- function(n_clusters, distance = "euclidean", linkage = "ward") {
  #This class is designed to perform agglomerative hierarchical clustering, 
  # a type of hierarchical clustering that builds a tree of clusters by iteratively merging the closest clusters.
  # parameters:
  # n_clusters (integer): The number of clusters to find. 
  # distance (string, default = "euclidean"): The metric used to compute the distance between observations. 
  #         Supported options are "euclidean", "l1", "l2", "manhattan", and "cosine".
  # linkage (string, default = "ward"): The linkage criterion determines which distance to use between sets of observations. 
  #         Supported options are "ward", "complete", "average", and "single".
  # example usage.
  # 
  # Create a kmeans object
  # ac <- agglo_clustering(n_clusters = 3, distance = "euclidean", linkage = "ward")
  # Generate some data
  # set.seed(123)
  # X <- matrix(rnorm(100), nrow = 10)
  # Fit the model
  # result <- fit(ac, X)
  # print(result$cluster_labels) 
  if (!distance %in% c("euclidean", "l1", "l2", "manhattan", "cosine")) {
    stop("Unsupported distance measure.")
  }
  if (!linkage %in% c("ward", "complete", "average", "single")) {
    stop("Unsupported linkage method.")
  }
  
  structure(list(n_clusters = n_clusters, distance = distance, linkage = linkage, cluster_labels = NULL), class = "agglo_clustering")
}

# Define the fit method
fit <- function(object, X) {
  if (!inherits(object, "agglo_clustering")) {
    stop("Object must be of class 'agglo_clustering'.")
  }
  
  # Compute distance matrix according to the specified metric
  dist_matrix <- dist(X, method = object$distance)
  
  # Perform hierarchical clustering
  hc <- hclust(dist_matrix, method = object$linkage)
  
  # Cut the tree to form clusters
  object$cluster_labels <- cutree(hc, k = object$n_clusters)
  
  # Return the updated object
  object
}

# Create an agglo_clustering object
ac <- agglo_clustering(n_clusters = 3, distance = "euclidean", linkage = "ward")

# Generate some data
set.seed(123)
X <- matrix(rnorm(100), nrow = 10)

# Fit the model
result <- fit(ac, X)

# Print results
print(result$cluster_labels)  # Cluster assignment for each example
