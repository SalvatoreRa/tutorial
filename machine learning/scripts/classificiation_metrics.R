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






