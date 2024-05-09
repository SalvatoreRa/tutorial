#################################################
##### Mini Batch K-means
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



