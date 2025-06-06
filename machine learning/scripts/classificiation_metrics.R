############################################
#### Accuracy score
###########################################

accuracy_score <- function(actual, predicted) {
  # Example data
  # actual_values <- c(0,0,0,1,1)
  # predicted_values <- c(0,1,0,1,1)
  # Calculate accuracy
  # accuracy <- accuracy_score(actual_values, predicted_values)
  # print(paste("Accuracy:", accuracy, "%"))
  # Ensure that actual and predicted are factors and have the same levels
  actual <- factor(actual, levels = unique(c(actual, predicted)))
  predicted <- factor(predicted, levels = levels(actual))
  
  # Calculate the accuracy as the proportion of correct predictions
  correct_predictions <- sum(actual == predicted)
  total_predictions <- length(actual)
  
  # Return the accuracy score as a percentage
  accuracy <- (correct_predictions / total_predictions) * 100
  return(accuracy)
}



############################################
#### Compute Area Under the Curve (AUC) 
###########################################

compute_auc <- function(true_values, predicted_probs) {
  # Compute Area Under the Curve (AUC) using the trapezoidal rule
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1)
  # predicted_probs <- c(0.1, 0.4, 0.35, 0.8)
  # Calculate AUC
  # auc_value <- compute_auc(true_values, predicted_probs)
  # print(paste("AUC:", auc_value))
  
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Prepare the data frame
  data <- data.frame(true_values = true_values, predicted_probs = predicted_probs)
  
  # Sort by predicted probabilities in descending order
  data <- data[order(-data$predicted_probs),]
  
  # Calculate the total number of positive and negative cases
  P <- sum(data$true_values)
  N <- nrow(data) - P
  
  # Initialize true positive and false positive counts
  tp <- 0
  fp <- 0
  
  # Initialize vectors to store FPR and TPR
  tpr <- c(0)
  fpr <- c(0)
  
  # Compute TPR and FPR for each threshold
  for (i in 1:nrow(data)) {
    if (data$true_values[i] == 1) {
      tp <- tp + 1
    } else {
      fp <- fp + 1
    }
    tpr <- c(tpr, tp / P)
    fpr <- c(fpr, fp / N)
  }
  
  # Append the endpoint (1,1)
  tpr <- c(tpr, 1)
  fpr <- c(fpr, 1)
  
  # Compute the AUC using the trapezoidal rule
  auc_value <- 0
  for (i in 2:length(tpr)) {
    base_width <- fpr[i] - fpr[i-1]
    average_height <- (tpr[i] + tpr[i-1]) / 2
    auc_value <- auc_value + (base_width * average_height)
  }
  
  return(auc_value)
}


############################################
#### Balanced accuracy
###########################################

balanced_accuracy <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 1, 0, 0, 1, 0)
  # predicted_values <- c(0, 1, 0, 0, 0, 1)
  # Calculate Balanced Accuracy
  # balanced_acc_value <- balanced_accuracy(true_values, predicted_values)
  # print(paste("Balanced Accuracy:", balanced_acc_value))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert true values and predicted values to factors to ensure that all levels are accounted for
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Calculate confusion matrix
  cm <- table(True = true_values, Predicted = predicted_values)
  
  # Calculate recall for each class
  recall_pos <- cm[2, 2] / sum(cm[2, ])  # True Positives / (True Positives + False Negatives)
  recall_neg <- cm[1, 1] / sum(cm[1, ])  # True Negatives / (True Negatives + False Positives)
  
  # Calculate balanced accuracy
  balanced_acc <- (recall_pos + recall_neg) / 2
  
  return(balanced_acc)
}

############################################
#### Brier score
###########################################

brier_score_loss <- function(true_values, predicted_probs) {
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 1, 1, 0)
  # predicted_probs <- c(0.1, 0.9, 0.8, 0.3)
  # Calculate Brier score loss
  # brier_score_value <- brier_score_loss(true_values, predicted_probs)
  # print(paste("Brier Score Loss:", brier_score_value))
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Ensure true values are either 0 or 1
  if (!all(true_values %in% c(0, 1))) {
    stop("True values should only contain 0s and 1s")
  }
  
  # Compute the Brier score loss
  n <- length(true_values)
  brier_score <- sum((predicted_probs - true_values)^2) / n
  
  return(brier_score)
}

############################################
#### Cohen Kappa
###########################################

cohen_kappa <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(2, 0, 2, 2, 0, 2)
  # predicted_values <- c(0, 0, 2, 2, 0, 2)
  # Calculate Cohen's Kappa
  # kappa_value <- cohen_kappa(true_values, predicted_values)
  # print(paste("Cohen's Kappa:", kappa_value))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  table <- table(true_values, predicted_values)
  
  # Total number of observations
  n <- sum(table)
  
  # Sum of products of marginals
  sum_prod_marginals <- sum(rowSums(table) * colSums(table))
  
  # Sum of squared table
  sum_squared_table <- sum(table^2)
  
  # Calculate observed agreement
  observed_agreement <- (n * sum_squared_table - sum_prod_marginals) / n
  
  # Calculate expected agreement
  expected_agreement <- (sum_prod_marginals - n^2) / n
  
  # Calculate Cohen's Kappa
  kappa <- (observed_agreement - expected_agreement) / (n^2 - expected_agreement)
  
  return(kappa)
}

############################################
#### Confusion matrix
###########################################

compute_confusion_matrix <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 0, 1, 1, 1, 0)
  # predicted_values <- c(0, 1, 1, 0, 1, 0)
  # Calculate the confusion matrix
  # conf_matrix <- compute_confusion_matrix(true_values, predicted_values)
  # print(conf_matrix)
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create the confusion matrix
  matrix <- table(Predicted = predicted_values, Actual = true_values)
  
  # Reorder the matrix to conventional format if necessary
  if (!identical(dimnames(matrix)$Predicted, c("0", "1"))) {
    matrix <- matrix[c("0", "1"), c("0", "1"), drop = FALSE]
  }
  
  return(matrix)
}


############################################
#### DET curve
#### Compute error rates for different probability thresholds.
###########################################

compute_error_rates <- function(true_values, predicted_probs, thresholds) {
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1)
  # predicted_probs <- c(0.1, 0.4, 0.35, 0.8)
  # Define a range of thresholds
  # thresholds <- seq(0, 1, by = 0.1)
  # Calculate error rates
  # error_rates <- compute_error_rates(true_values, predicted_probs, thresholds)
  # Plotting the DET curve using base R plotting
  # plot(error_rates$false_acceptance_rates, error_rates$false_rejection_rates, type = "b",
  #     xlab = "False Acceptance Rate", ylab = "False Rejection Rate",
  #     main = "DET Curve")
  # points(error_rates$false_acceptance_rates, error_rates$false_rejection_rates, pch = 19, col = "red")
  
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Initialize vectors to store error rates
  false_acceptance_rates <- numeric(length(thresholds))
  false_rejection_rates <- numeric(length(thresholds))
  
  # Compute error rates for each threshold
  for (i in seq_along(thresholds)) {
    threshold <- thresholds[i]
    

    predicted_labels <- ifelse(predicted_probs >= threshold, 1, 0)

    false_acceptances <- sum((predicted_labels == 1) & (true_values == 0))
    false_rejections <- sum((predicted_labels == 0) & (true_values == 1))

    total_negatives <- sum(true_values == 0)
    total_positives <- sum(true_values == 1)
    
   
    false_acceptance_rates[i] <- false_acceptances / total_negatives
    false_rejection_rates[i] <- false_rejections / total_positives
  }
  
  return(list(thresholds = thresholds, false_acceptance_rates = false_acceptance_rates, false_rejection_rates = false_rejection_rates))
}

############################################
#### F1 score
###########################################

compute_f1_score <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 1, 1, 1)
  # predicted_values <- c(0, 1, 1, 1)
  # Calculate the F1 Score
  # f1_score <- compute_f1_score(true_values, predicted_values)
  # print(paste("F1 Score:", f1_score))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Ensure all elements exist in the matrix to prevent errors in the calculations
  if (!all(c("0", "1") %in% rownames(cm))) {
    cm <- addmargins(cm)
  }
  if (!all(c("0", "1") %in% colnames(cm))) {
    cm <- addmargins(cm)
  }

  precision <- cm[2, 2] / sum(cm[2, ])
  recall <- cm[2, 2] / sum(cm[, 2])
  if (precision + recall == 0) {
    return(0)
  }
  
  # Calculate F1 Score
  f1_score <- 2 * ((precision * recall) / (precision + recall))
  
  return(f1_score)
}

############################################
#### F-beta score
###########################################

compute_fbeta_score <- function(true_values, predicted_values, beta) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 1, 1, 1)
  # predicted_values <- c(0, 1, 1, 1)
  # beta_value <- 2  # Emphasizing recall
  # Calculate the F-beta Score
  # fbeta_score <- compute_fbeta_score(true_values, predicted_values, beta_value)
  # print(paste("F-beta Score:", fbeta_score))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  if (beta < 0) {
    stop("Beta should be non-negative")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Ensure all elements exist in the matrix to prevent errors in the calculations
  if (!all(c("0", "1") %in% rownames(cm))) {
    cm <- addmargins(cm)
  }
  if (!all(c("0", "1") %in% colnames(cm))) {
    cm <- addmargins(cm)
  }
  
  # Calculate Precision and Recall
  precision <- cm[2, 2] / sum(cm[2, ])
  recall <- cm[2, 2] / sum(cm[, 2])
  
  # Handle case where precision and recall are both zero
  if (precision + recall == 0) {
    return(0)
  }
  
  # Calculate F-beta Score
  fbeta_score <- (1 + beta^2) * ((precision * recall) / ((beta^2 * precision) + recall))
  
  return(fbeta_score)
}


############################################
#### Jaccard score
###########################################

compute_jaccard_score <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 1, 1, 1)
  # predicted_values <- c(0, 1, 1, 1)
  # Calculate the Jaccard Score
  # jaccard_score <- compute_jaccard_score(true_values, predicted_values)
  # print(paste("Jaccard Score:", jaccard_score))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Ensure all elements exist in the matrix to prevent errors in the calculations
  if (!all(c("0", "1") %in% rownames(cm))) {
    cm <- addmargins(cm)
  }
  if (!all(c("0", "1") %in% colnames(cm))) {
    cm <- addmargins(cm)
  }
  
  # Calculate intersection (true positives) and union (true positives + false positives + false negatives)
  true_positives <- cm[2, 2]
  false_positives <- cm[2, 1]
  false_negatives <- cm[1, 2]
  
  intersection_size <- true_positives
  union_size <- true_positives + false_positives + false_negatives
  
  # Calculate Jaccard score
  if (union_size == 0) {
    return(0)  # Handle the case where there are no positive predictions or actuals
  }
  
  jaccard_score <- intersection_size / union_size
  
  return(jaccard_score)
}


############################################
#### Log loss or cross entropy loss
###########################################

compute_cross_entropy_loss <- function(true_values, predicted_probs) {
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1)
  # predicted_probs <- c(0.05, 0.1, 0.8, 0.95)
  # Calculate the Cross-Entropy Loss
  # cross_entropy_loss <- compute_cross_entropy_loss(true_values, predicted_probs)
  # print(paste("Cross-Entropy Loss:", cross_entropy_loss))
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Ensure all probabilities are valid
  if (any(predicted_probs < 0 | predicted_probs > 1)) {
    stop("Predicted probabilities must be between 0 and 1")
  }
  
  # Compute the cross-entropy loss
  epsilon <- 1e-15  # A small number to prevent log(0)
  predicted_probs <- pmax(pmin(predicted_probs, 1 - epsilon), epsilon)  # Clipping for stability in log calculations
  loss <- -mean(true_values * log(predicted_probs) + (1 - true_values) * log(1 - predicted_probs))
  
  return(loss)
}


############################################
#### Matthews correlation coefficient (MCC)
###########################################

compute_mcc <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(+1, +1, +1, -1)
  # predicted_values <- c(+1, -1, +1, +1)
  # mcc_value <- compute_mcc(true_values, predicted_values)
  # print(paste("Matthews Correlation Coefficient (MCC):", mcc_value))
  
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert true_values and predicted_values to factor ensuring both have two levels: 1 and -1
  true_values <- factor(true_values, levels = c(-1, 1))
  predicted_values <- factor(predicted_values, levels = c(-1, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Extract confusion matrix components
  tp <- ifelse(!is.na(cm["1", "1"]), cm["1", "1"], 0)
  tn <- ifelse(!is.na(cm["-1", "-1"]), cm["-1", "-1"], 0)
  fp <- ifelse(!is.na(cm["1", "-1"]), cm["1", "-1"], 0)
  fn <- ifelse(!is.na(cm["-1", "1"]), cm["-1", "1"], 0)
  
  # Compute Matthews correlation coefficient
  numerator <- (tp * tn) - (fp * fn)
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  # Handle cases where denominator is 0 to avoid division by zero
  if (denominator == 0) {
    return(0)
  }
  
  mcc <- numerator / denominator
  
  return(mcc)
}


############################################
#### precision-recall curve
###########################################

compute_precision_recall_curve <- function(true_values, predicted_probs) {
  
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1, 1)
  # predicted_probs <- c(0.1, 0.2, 0.8, 0.4, 0.95)
  # Compute the Precision-Recall Curve
  # pr_curve <- compute_precision_recall_curve(true_values, predicted_probs)
  # Plot the Precision-Recall Curve using base R plotting
  # plot(pr_curve$recalls, pr_curve$precisions, type = 'b', col = 'blue',
  #     xlab = 'Recall', ylab = 'Precision', main = 'Precision-Recall Curve')
  # points(pr_curve$recalls, pr_curve$precisions, pch = 19, col = 'red')
  
  
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Combine true values and predicted probabilities in a data frame, sort by predicted_probs descending
  data <- data.frame(true_values, predicted_probs)
  data <- data[order(-data$predicted_probs),]
  
  # Initialize vectors to store precision and recall values
  precisions <- numeric()
  recalls <- numeric()
  thresholds <- unique(data$predicted_probs)
  
  # Calculate precision and recall for each threshold
  for (threshold in thresholds) {
    # Classify predictions based on current threshold
    predicted_values <- ifelse(data$predicted_probs >= threshold, 1, 0)
    
    # Calculate confusion matrix components
    tp <- sum(predicted_values == 1 & data$true_values == 1)
    fp <- sum(predicted_values == 1 & data$true_values == 0)
    fn <- sum(predicted_values == 0 & data$true_values == 1)
    
    # Calculate precision and recall
    precision <- if (tp + fp == 0) 1 else tp / (tp + fp)
    recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
    
    # Store results
    precisions <- c(precisions, precision)
    recalls <- c(recalls, recall)
  }
  
  return(list(precisions = precisions, recalls = recalls, thresholds = thresholds))
}


############################################
#### precision score
###########################################

compute_precision <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 0, 1, 1)
  # predicted_values <- c(0, 1, 1, 1)
  # Calculate Precision
  # precision_value <- compute_precision(true_values, predicted_values)
  # print(paste("Precision Score:", precision_value))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Ensure all elements exist in the confusion matrix
  if (!all(c("0", "1") %in% rownames(cm))) {
    cm <- addmargins(cm)
  }
  if (!all(c("0", "1") %in% colnames(cm))) {
    cm <- addmargins(cm)
  }
  
  # Extract confusion matrix components
  tp <- ifelse(!is.na(cm["1", "1"]), cm["1", "1"], 0)
  fp <- ifelse(!is.na(cm["1", "0"]), cm["1", "0"], 0)
  
  # Calculate Precision
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  
  return(precision)
}




############################################
#### recall score
###########################################

compute_recall <- function(true_values, predicted_values) {
  # Example true binary outcomes and predicted class labels
  # true_values <- c(0, 0, 1, 1)
  # predicted_values <- c(0, 1, 0, 1)
  # Calculate Recall
  # recall_value <- compute_recall(true_values, predicted_values)
  # print(paste("Recall Score:", recall_value))
  if (length(true_values) != length(predicted_values)) {
    stop("True values and predicted values must have the same length")
  }
  
  # Convert to factors to ensure all possible outcomes (0 and 1) are included
  true_values <- factor(true_values, levels = c(0, 1))
  predicted_values <- factor(predicted_values, levels = c(0, 1))
  
  # Create a confusion matrix
  cm <- table(Predicted = predicted_values, Actual = true_values)
  
  # Ensure all elements exist in the confusion matrix
  if (!all(c("0", "1") %in% rownames(cm))) {
    cm <- addmargins(cm)
  }
  if (!all(c("0", "1") %in% colnames(cm))) {
    cm <- addmargins(cm)
  }
  
  # Extract confusion matrix components
  tp <- ifelse(!is.na(cm["1", "1"]), cm["1", "1"], 0)
  fn <- ifelse(!is.na(cm["0", "1"]), cm["0", "1"], 0)
  
  # Calculate Recall
  recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
  
  return(recall)
}

############################################
#### AUROC score
###########################################

compute_roc_auc <- function(true_values, predicted_probs) {
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1)
  # predicted_probs <- c(0., 0., 0.8, 0.9)
  # Calculate the ROC AUC
  # roc_auc_value <- compute_roc_auc(true_values, predicted_probs)
  # print(paste("ROC AUC:", roc_auc_value))
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Create a data frame of true values and predicted probabilities
  data <- data.frame(true_values = true_values, predicted_probs = predicted_probs)
  
  # Order by predicted probabilities in descending order
  data <- data[order(-data$predicted_probs),]
  
  # Append true positive and false positive rates
  data$tp = cumsum(data$true_values == 1)
  data$fp = cumsum(data$true_values == 0)
  
  # Calculate total positives and negatives
  n_pos <- max(data$tp)
  n_neg <- max(data$fp)
  
  # Normalize tp and fp by total positives and negatives
  data$tpr = data$tp / n_pos
  data$fpr = data$fp / n_neg
  
  # Calculate ROC AUC using the trapezoidal rule
  roc_auc <- sum(
    with(data, diff(c(0, fpr)) * (tpr[-length(tpr)] + tpr[-1]) / 2)
  )
  
  return(roc_auc)
}




############################################
#### AUC curve
###########################################


plot_roc_curve <- function(true_values, predicted_probs) {
  # Example true binary outcomes and predicted probabilities
  # true_values <- c(0, 0, 1, 1)
  # predicted_probs <- c(0.1, 0.2, 0.8, 0.9)
  # Plot the ROC Curve and display the AUC
  # auc_value <- plot_roc_curve(true_values, predicted_probs)
  # print(paste("Area Under Curve (AUC):", format(auc_value, digits = 4)))
  
  if (length(true_values) != length(predicted_probs)) {
    stop("True values and predicted probabilities must have the same length")
  }
  
  # Create a data frame of true values and predicted probabilities
  data <- data.frame(true_values = true_values, predicted_probs = predicted_probs)
  
  # Order by predicted probabilities in descending order
  data <- data[order(-data$predicted_probs),]
  
  # Append true positive and false positive rates
  data$tp = cumsum(data$true_values == 1)
  data$fp = cumsum(data$true_values == 0)
  
  # Calculate total positives and negatives
  n_pos <- max(data$tp)
  n_neg <- max(data$fp)
  
  # Normalize tp and fp by total positives and negatives
  tpr = data$tp / n_pos
  fpr = data$fp / n_neg
  
  # Add (0,0) and (1,1) to the curve
  tpr = c(0, tpr, 1)
  fpr = c(0, fpr, 1)
  
  # Calculate AUC using the trapezoidal rule
  auc <- sum(diff(fpr) * (tpr[-length(tpr)] + tpr[-1]) / 2)
  
  # Plot the ROC Curve
  plot(fpr, tpr, type = "l", col = "red", lwd = 2,
       xlab = "False Positive Rate (1 - Specificity)",
       ylab = "True Positive Rate (Sensitivity)",
       main = paste("ROC Curve (AUC = ", format(round(auc, 2), nsmall = 2), ")", sep = ""))
  grid()
  abline(a = 0, b = 1, lty = 2, col = "blue")  # Add reference line
  
  return(auc)
}





