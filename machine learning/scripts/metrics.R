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



