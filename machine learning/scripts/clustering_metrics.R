#################################################
##### Adjusted Mutual Information between two clusterings.
#####
#################################################

# Function to calculate entropy
entropy <- function(clustering) {
  n <- length(clustering)
  proportions <- table(clustering) / n
  -sum(proportions * log(proportions))
}

# Function to calculate mutual information
mutual_information <- function(cluster1, cluster2) {
  n <- length(cluster1)
  contingency_table <- table(cluster1, cluster2)
  joint_prob <- contingency_table / n
  marginal1 <- margin.table(contingency_table, 1) / n
  marginal2 <- margin.table(contingency_table, 2) / n
  
  mutual_info <- 0
  for (i in seq_len(nrow(contingency_table))) {
    for (j in seq_len(ncol(contingency_table))) {
      if (contingency_table[i, j] > 0) {
        mutual_info <- mutual_info + joint_prob[i, j] * log(joint_prob[i, j] / (marginal1[i] * marginal2[j]))
      }
    }
  }
  mutual_info
}

# Function to calculate expected mutual information
expected_mi <- function(cluster1, cluster2) {
  n <- length(cluster1)
  contingency_table <- table(cluster1, cluster2)
  marginal1 <- margin.table(contingency_table, 1)
  marginal2 <- margin.table(contingency_table, 2)
  
  emi <- 0
  for (i in seq_len(nrow(contingency_table))) {
    for (j in seq_len(ncol(contingency_table))) {
      expected_count <- marginal1[i] * marginal2[j] / n
      if (expected_count > 0) {
        emi <- emi + expected_count / n * log(expected_count / n / (marginal1[i] / n * marginal2[j] / n))
      }
    }
  }
  emi
}

# Function to calculate adjusted mutual information
adjusted_mutual_information <- function(cluster1, cluster2) {
  # Adjusted Mutual Information between two clusterings results.
  # argumenrs: two cluster list
  # Example of usage
  # cluster1 <- sample(1:3, 100, replace = TRUE)
  # cluster2 <- sample(1:3, 100, replace = TRUE)
  # adjusted_mutual_information_result <- adjusted_mutual_information(cluster1, cluster2)
  # print(adjusted_mutual_information_result)
  mi <- mutual_information(cluster1, cluster2)
  expected_mi_val <- expected_mi(cluster1, cluster2)
  max_entropy <- max(entropy(cluster1), entropy(cluster2))
  
  if (max_entropy == 0) {
    return(1)
  } else {
    ami <- (mi - expected_mi_val) / (0.5 * (entropy(cluster1) + entropy(cluster2)) - expected_mi_val)
    return(ami)
  }
}

#################################################
##### Rand index adjusted for chance.
#####
#################################################

# Adjusted Rand Index
adjusted_rand_index <- function(cluster1, cluster2) {
  # Adjusted Rand Index
  #
  # argumenrs: two cluster list
  # Example of usage
  # cluster1 <- sample(1:3, 100, replace = TRUE)
  # cluster2 <- sample(1:3, 100, replace = TRUE)
  # ari_result  <- adjusted_rand_index(cluster1, cluster2)
  # print(ari_result )
  contingency_table <- table(cluster1, cluster2)
  
  # Calculate sums over rows and columns
  sum_rows <- rowSums(contingency_table)
  sum_cols <- colSums(contingency_table)
  
  # Total sum of the contingency table (total pairs)
  n <- sum(contingency_table)
  
  # Sum of combinations of the sums of rows and columns
  sum_row_comb <- sum(sum_rows * (sum_rows - 1) / 2)
  sum_col_comb <- sum(sum_cols * (sum_cols - 1) / 2)
  
  # Calculate sum of combinations for each cell in the contingency table
  sum_comb_nij <- sum(contingency_table * (contingency_table - 1) / 2)
  
  # Calculate Rand index components
  total_combinations <- n * (n - 1) / 2
  index <- sum_comb_nij
  expected_index <- sum_row_comb * sum_col_comb / total_combinations
  max_index <- (sum_row_comb + sum_col_comb) / 2
  
  # Adjusted Rand Index
  ari <- (index - expected_index) / (max_index - expected_index)
  return(ari)
}



#################################################
#####  the Calinski and Harabasz score.
#####
#################################################


calinski_harabasz_score <- function(X, labels) {
  # Function to compute Calinski-Harabasz score
  # 
  # parameters:
  # the X matrix
  # cluster list
  # # Example usage
  # set.seed(123)
  # X <- matrix(rnorm(200), nrow = 50, ncol = 2)
  # labels <- sample(1:4, 50, replace = TRUE)
  # ch_score <- calinski_harabasz_score(X, labels)
  # print(ch_score)
  if (!is.matrix(X)) {
    stop("Input data X must be a matrix.")
  }
  
  # Number of samples and clusters
  n <- nrow(X)
  k <- length(unique(labels))
  if (k == 1 || k == n) {
    stop("k must be greater than 1 and less than n for CH index calculation.")
  }
  
  # Calculate the overall mean
  overall_mean <- colMeans(X)
  
  # Calculate the between group dispersion matrix (Bk)
  Bk <- matrix(0, ncol = ncol(X), nrow = ncol(X))
  cluster_means <- tapply(seq(n), labels, function(i) colMeans(X[i, , drop = FALSE]))
  cluster_sizes <- table(labels)
  
  for (cl in names(cluster_sizes)) {
    n_cl <- cluster_sizes[[cl]]
    mean_cl <- cluster_means[[cl]]
    Bk <- Bk + n_cl * (mean_cl - overall_mean) %*% t(mean_cl - overall_mean)
  }
  trace_Bk <- sum(diag(Bk))
  
  # Calculate the within-cluster dispersion matrix (Wk)
  Wk <- matrix(0, ncol = ncol(X), nrow = ncol(X))
  for (cl in names(cluster_sizes)) {
    points_in_cluster <- X[labels == cl, , drop = FALSE]
    mean_cl <- cluster_means[[cl]]
    deviations <- sweep(points_in_cluster, 2, mean_cl)
    Wk <- Wk + t(deviations) %*% deviations
  }
  trace_Wk <- sum(diag(Wk))
  
  # Calculate the CH index
  CH <- (trace_Bk / (k - 1)) / (trace_Wk / (n - k))
  return(CH)
}


#################################################
#####  the Calinski and Harabasz score.
#####
#################################################

# Function to calculate the Davies-Bouldin index
davies_bouldin_score <- function(X, labels) {
  # Function to compute the Davies-Bouldin score.
  # The minimum score is zero, with lower values indicating better clustering.
  
  # parameters:
  # the X matrix
  # cluster list labels
  
  # Example usage
  # set.seed(123)
  # X <- matrix(rnorm(200), nrow = 50, ncol = 2)
  # labels <- sample(1:3, 50, replace = TRUE)
  # db_score <- davies_bouldin_score(X, labels)
  # print(db_score)
  
  if (!is.matrix(X)) {
    stop("Input data X must be a matrix.")
  }
  
  # Unique clusters
  clusters <- unique(labels)
  k <- length(clusters)
  if (k < 2) {
    stop("There must be at least two clusters to calculate the Davies-Bouldin score.")
  }
  
  # Calculate centroids and dispersions
  centroids <- tapply(seq(nrow(X)), labels, function(indices) colMeans(X[indices, ]))
  dispersions <- numeric(k)
  
  for (i in seq_along(clusters)) {
    cluster_points <- X[labels == clusters[i], , drop = FALSE]
    dispersions[i] <- mean(sqrt(rowSums((cluster_points - centroids[[i]])^2)))
  }
  
  # Calculate Davies-Bouldin index
  db_index <- numeric(k)
  
  for (i in seq_along(clusters)) {
    max_ratio <- 0
    for (j in seq_along(clusters)) {
      if (i != j) {
        numerator <- dispersions[i] + dispersions[j]
        denominator <- sqrt(sum((centroids[[i]] - centroids[[j]])^2))
        ratio <- numerator / denominator
        if (ratio > max_ratio) {
          max_ratio <- ratio
        }
      }
    }
    db_index[i] <- max_ratio
  }
  
  # Return the average of the maximum ratios
  mean(db_index)
}


#################################################
#####  Compute completeness metric of a cluster labeling given a ground truth.
##### 
##### A clustering result satisfies completeness if all the data points 
##### that are members of a given class are elements of the same cluster.
#####
#################################################


conditional_entropy <- function(classes, clusters) {
  unique_classes <- unique(classes)
  unique_clusters <- unique(clusters)
  n <- length(classes)
  
  # Calculate the joint probability distribution and the marginal probability of clusters
  joint_prob <- table(classes, clusters) / n
  cluster_prob <- table(clusters) / n
  
  # Calculate conditional entropy
  cond_entropy <- 0
  for (cluster in unique_clusters) {
    cluster_cond_prob <- joint_prob[, cluster] / cluster_prob[cluster]
    cluster_cond_prob <- cluster_cond_prob[cluster_cond_prob > 0]  # Avoid NaN for log(0)
    cond_entropy <- cond_entropy + cluster_prob[cluster] * sum(-cluster_cond_prob * log(cluster_cond_prob))
  }
  
  cond_entropy
}


completeness_score <- function(true_labels, cluster_labels) {
  # Function to Compute completeness metric of a cluster labeling given a ground truth.
  # Score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling.
  
  # parameters:
  # true labels
  # cluster list labels
  
  # Example usage
  # set.seed(123)
  # true_labels <- c(1, 1, 1, 2, 2, 2, 3, 3, 3)
  #cluster_labels <- c(1, 1, 1, 2, 2, 2, 1, 1, 1)
  # completeness <- completeness_score(true_labels, cluster_labels)
  # print(completeness)
  h_g <- entropy(true_labels)
  h_g_given_c <- conditional_entropy(true_labels, cluster_labels)
  
  # Calculate completeness
  if (h_g == 0) return(1)  # Perfect completeness when there is no entropy in the labels
  1 - h_g_given_c / h_g
}


#################################################
##### a contingency matrix describing the relationship between labels.
#####
#################################################


build_contingency_matrix <- function(labels1, labels2) {
  # build a contingency matrix describing the relationship between labels.
  # 
  # parameters:
  # true labels or another cluster label list
  # cluster list labels
  
  # Example usage
  # labels1 <- sample(1:3, 20, replace = TRUE)  # Sample clustering results
  # labels2 <- sample(1:4, 20, replace = TRUE)  # Sample clustering or true labels
  # contingency_matrix <- build_contingency_matrix(labels1, labels2)
  # print(contingency_matrix)
  if (length(labels1) != length(labels2)) {
    stop("Both label lists must be of the same length.")
  }
  
  # Create a table (contingency matrix) of the labels
  contingency_table <- table(labels1, labels2)
  
  # Optionally, you can add names to the dimensions for clarity
  dimnames(contingency_table) <- list(Cluster_Group_1 = rownames(contingency_table),
                                      Cluster_Group_2 = colnames(contingency_table))
  
  return(contingency_table)
}




#################################################
#####  calculate the Fowlkes-Mallows
##### Measure the similarity of two clusterings of a set of points.
#################################################

calculate_pairs <- function(labels_true, labels_pred) {
  # calculate the Fowlkes-Mallows
  # 
  # parameters:
  # true labels or another cluster label list
  # cluster list labels
  #
  # # Example usage
  # labels_true <- c(1, 1, 2, 2, 3, 3)
  # labels_pred <- c(1, 1, 1, 2, 3, 3)
  #  fms <- fowlkes_mallows_score(labels_true, labels_pred)
  # print(fms)
  if (length(labels_true) != length(labels_pred)) {
    stop("Both label lists must be of the same length.")
  }
  
  # Initialize counts
  TP <- FP <- TN <- FN <- 0
  n <- length(labels_true)
  
  # Compare each pair of elements
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      true_same <- labels_true[i] == labels_true[j]
      pred_same <- labels_pred[i] == labels_pred[j]
      
      if (true_same && pred_same) {
        TP <- TP + 1  # True Positive
      } else if (!true_same && pred_same) {
        FP <- FP + 1  # False Positive
      } else if (true_same && !pred_same) {
        FN <- FN + 1  # False Negative
      }
    }
  }
  
  list(TP = TP, FP = FP, FN = FN)
}

# Function to calculate Fowlkes-Mallows score
fowlkes_mallows_score <- function(labels_true, labels_pred) {
  pairs <- calculate_pairs(labels_true, labels_pred)
  TP <- pairs$TP
  FP <- pairs$FP
  FN <- pairs$FN
  
  # Compute the Fowlkes-Mallows score
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  FMS <- sqrt(precision * recall)
  
  return(FMS)
}


#################################################
#####  Homogeneity score
##### 
#################################################

# Function to calculate entropy
entropy <- function(labels) {
  if (length(labels) == 0) return(0)
  probs <- table(labels) / length(labels)
  -sum(probs * log(probs))
}

# Function to calculate conditional entropy H(C|K)
conditional_entropy_sum <- function(classes, clusters) {
  unique_clusters <- unique(clusters)
  total_samples <- length(classes)
  sum_conditional_entropy <- 0
  
  for (cluster in unique_clusters) {
    cluster_indices <- which(clusters == cluster)
    cluster_classes <- classes[cluster_indices]
    cluster_size <- length(cluster_indices)
    
    # Entropy of the classes within the cluster
    entropy_cluster <- entropy(cluster_classes)
    sum_conditional_entropy <- sum_conditional_entropy + (cluster_size / total_samples) * entropy_cluster
  }
  
  sum_conditional_entropy
}


homogeneity_score <- function(true_labels, cluster_labels) {
  # Homogeneity metric of a cluster labeling given a ground truth.
  # A clustering result satisfies homogeneity if all of its clusters contain 
  # only data points which are members of a single class.
  # parameters:
  # true labels or another cluster label list
  # cluster list labesls
  #
  # # Example usage
  # true_labels <- c(1, 1, 1, 2, 2, 2, 3, 3, 3)
  # cluster_labels <- c(1, 1, 1, 1, 2, 2, 3, 3, 3)
  #homogeneity <- homogeneity_score(true_labels, cluster_labels)
  # print(homogeneity)
  total_entropy <- entropy(true_labels)
  if (total_entropy == 0) return(1)  # Perfect homogeneity if there's no entropy in true labels
  
  conditional_entropy_value <- conditional_entropy_sum(true_labels, cluster_labels)
  homogeneity <- 1 - (conditional_entropy_value / total_entropy)
  return(homogeneity)
}



#################################################
#####  silhouette scores
##### 
#################################################



euclidean_distance_points <- function(point1, point2) {
  sqrt(sum((point1 - point2)^2))
}


average_distance_points <- function(point, others) {
  sum(sapply(1:nrow(others), function(i) euclidean_distance_points(point, others[i,]))) / nrow(others)
}


silhouette_scores <- function(X, clusters) {
  # calculate silhouette score
  # Example usage:
  # X <- matrix(runif(50), ncol=5)
  # clusters <- sample(1:3, 10, replace = TRUE)
  # silhouette_result <- silhouette_scores(X, clusters)
 # print(silhouette_result)
  unique_clusters <- unique(clusters)
  silhouette_values <- numeric(length(clusters))
  
  # Loop over all points
  for (i in 1:nrow(X)) {
    cluster <- clusters[i]
    own_cluster_points <- X[clusters == cluster, , drop = FALSE]
    
    # Average intra-cluster distance
    a_i <- average_distance_points(X[i, ], own_cluster_points[-i, , drop = FALSE])
    
    # Minimum average inter-cluster distance to the nearest cluster
    b_i <- Inf
    for (other_cluster in unique_clusters[unique_clusters != cluster]) {
      other_cluster_points <- X[clusters == other_cluster, , drop = FALSE]
      b_i <- min(b_i, average_distance(X[i, ], other_cluster_points))
    }
    
    # Compute the silhouette value for the point
    silhouette_values[i] <- (b_i - a_i) / max(a_i, b_i)
  }
  
  return(mean(silhouette_values, na.rm = TRUE))
}




