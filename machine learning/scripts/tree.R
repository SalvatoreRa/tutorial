############################################
#### Decision tree classifier
###########################################

DecisionTreeClassifier <- setRefClass(
  "DecisionTreeClassifier",
  # # Generate some example data
  # set.seed(123)
  # X <- matrix(rnorm(200), nrow = 100, ncol = 2)
  # y <- ifelse(X[, 1] + X[, 2] > 0, 1, 0)  
  # Create and fit the decision tree model
  # tree_model <- DecisionTreeClassifier$new(max_depth = 3)
  # tree_model$fit(X, y)
  # Predict using the fitted model
  # predictions <- tree_model$predict(X)
  # print("Predictions:")
  # print(predictions)
  fields = list(
    max_depth = "numeric",  # Maximum depth of the tree
    tree = "list"           # The structure of the tree, initialized as an empty list
  ),
  
  methods = list(
    initialize = function(max_depth = 5) {
      .self$max_depth <- max_depth
      .self$tree <- list()  # Ensure this is initialized as an empty list, not NULL
    },
    
    fit = function(X, y) {
      # Validate inputs
      if (!is.matrix(X)) {
        stop("Input X must be a matrix.")
      }
      if (!all(y %in% c(0, 1))) {
        stop("Input y must be a binary vector with values 0 and 1.")
      }
      
      # Convert y to factor and ensure levels are correctly set as "0" and "1"
      y <- factor(y, levels = c(0, 1))
      
      # Start building the tree
      .self$tree <- recursive_split(X, y, depth = 0)
    },
    
    recursive_split = function(X, y, depth) {
      # Handle case when there's no data to split
      if (nrow(X) == 0) {
        return(list(is_leaf = TRUE, class = NA))
      }
      # Determine stopping conditions
      if (depth >= .self$max_depth || nrow(X) < 2 || length(unique(y)) == 1) {
        # Return the mode of y as class
        return(list(is_leaf = TRUE, class = as.numeric(levels(y))[which.max(table(y))]))
      }
      
      # Initialize variables for the best split
      best_feature <- NULL
      best_threshold <- NULL
      best_score <- Inf
      
      # Iterate through each feature
      for (feature in seq(ncol(X))) {
        unique_values <- sort(unique(X[, feature]))
        thresholds <- head(unique_values, -1) + diff(unique_values) / 2
        
        for (threshold in thresholds) {
          left_idx <- X[, feature] <= threshold
          right_idx <- !left_idx
          left_y <- y[left_idx]
          right_y <- y[right_idx]
          score <- (length(left_y) / length(y)) * gini(left_y) +
            (length(right_y) / length(y)) * gini(right_y)
          
          if (score < best_score) {
            best_score <- score
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
      
      # Check if a valid split was found
      if (!is.null(best_feature)) {
        left_idx <- X[, best_feature] <= best_threshold
        right_idx <- !left_idx
        left_tree <- recursive_split(X[left_idx, , drop = FALSE], y[left_idx], depth + 1)
        right_tree <- recursive_split(X[right_idx, , drop = FALSE], y[right_idx], depth + 1)
        
        return(list(is_leaf = FALSE, feature = best_feature, threshold = best_threshold, left = left_tree, right = right_tree))
      } else {
        # No valid split found, return a leaf node
        return(list(is_leaf = TRUE, class = as.numeric(levels(y))[which.max(table(y))]))
      }
    },
    
    predict = function(X) {
      if (is.null(.self$tree)) {
        stop("The model has not been fitted yet.")
      }
      apply(X, 1, function(x) predict_instance(x, .self$tree))
    },
    
    predict_instance = function(x, node) {
      if (node$is_leaf) {
        return(node$class)
      } else {
        if (x[node$feature] <= node$threshold) {
          return(predict_instance(x, node$left))
        } else {
          return(predict_instance(x, node$right))
        }
      }
    },
    
    gini = function(y) {
      p <- table(y) / length(y)
      return(1 - sum(p^2))
    }
  )
)

############################################
#### Decision tree regressor
###########################################

DecisionTreeRegressor <- setRefClass(
  "DecisionTreeRegressor",
  # Generate some example data
  # set.seed(123)
  # X <- matrix(rnorm(200), nrow = 100, ncol = 2)
  # y <- X[, 1] * 2.5 + X[, 2] * -1.5 + rnorm(100)  
  # Create and fit the decision tree regressor
  # tree_regressor <- DecisionTreeRegressor$new(max_depth = 3)
  # tree_regressor$fit(X, y)
  # Predict using the fitted model
  # predictions <- tree_regressor$predict(X)
  # print("Predictions:")
  # print(predictions)
  
  fields = list(
    max_depth = "numeric",  # Maximum depth of the tree
    tree = "list"           # The structure of the tree
  ),
  
  methods = list(
    initialize = function(max_depth = 5) {
      .self$max_depth <- max_depth
      .self$tree <- list()  # Ensure this is initialized as an empty list, not NULL
    },
    
    fit = function(X, y) {
      # Validate inputs
      if (!is.matrix(X)) {
        stop("Input X must be a matrix.")
      }
      if (!is.numeric(y)) {
        stop("Input y must be a numeric vector.")
      }
      
      # Start building the tree
      .self$tree <- recursive_split(X, y, depth = 0)
    },
    
    recursive_split = function(X, y, depth) {
      # Check for stopping conditions
      if (depth >= .self$max_depth || nrow(X) < 2) {
        return(list(is_leaf = TRUE, value = mean(y)))
      }
      
      # Initialize variables for the best split
      best_feature <- NULL
      best_threshold <- NULL
      best_score <- Inf
      
      # Iterate through each feature
      for (feature in seq(ncol(X))) {
        unique_values <- sort(unique(X[, feature]))
        thresholds <- head(unique_values, -1) + diff(unique_values) / 2
        
        for (threshold in thresholds) {
          left_idx <- X[, feature] <= threshold
          right_idx <- !left_idx
          left_y <- y[left_idx]
          right_y <- y[right_idx]
          score <- mse_split(left_y, right_y)
          
          if (score < best_score) {
            best_score <- score
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
      
      # Check if a valid split was found
      if (!is.null(best_feature)) {
        left_idx <- X[, best_feature] <= best_threshold
        right_idx <- !left_idx
        left_tree <- recursive_split(X[left_idx, , drop = FALSE], y[left_idx], depth + 1)
        right_tree <- recursive_split(X[right_idx, , drop = FALSE], y[right_idx], depth + 1)
        
        return(list(is_leaf = FALSE, feature = best_feature, threshold = best_threshold, left = left_tree, right = right_tree))
      } else {
        return(list(is_leaf = TRUE, value = mean(y)))
      }
    },
    
    predict = function(X) {
      if (is.null(.self$tree)) {
        stop("The model has not been fitted yet.")
      }
      apply(X, 1, function(x) predict_instance(x, .self$tree))
    },
    
    predict_instance = function(x, node) {
      if (node$is_leaf) {
        return(node$value)
      } else {
        if (x[node$feature] <= node$threshold) {
          return(predict_instance(x, node$left))
        } else {
          return(predict_instance(x, node$right))
        }
      }
    },
    
    mse_split = function(left_y, right_y) {
      n_left <- length(left_y)
      n_right <- length(right_y)
      if (n_left == 0 || n_right == 0) return(Inf)
      
      mean_left <- mean(left_y)
      mean_right <- mean(right_y)
      
      mse_left <- sum((left_y - mean_left)^2)
      mse_right <- sum((right_y - mean_right)^2)
      
      return((mse_left + mse_right) / (n_left + n_right))
    }
  )
)

############################################
#### Random Forest Classifier
###########################################

RandomForestClassifier <- setRefClass(
  # classifier <- RandomForestClassifier$new()
  # Data simulation
  # set.seed(123)
  # X <- matrix(rnorm(100 * 4), ncol = 4)
  # y <- sample(0:1, 100, replace = TRUE)
  # Fit the model
  # classifier$fit(X, y)
  # Predict
  # predictions <- classifier$predict(X)
  # Get feature importances
  # importances <- classifier$getFeatureImportances()
  # print(predictions)
  # print(importances)
  "RandomForestClassifier",
  fields = list(
    trees = "list",
    num_trees = "numeric",
    feature_importances_ = "matrix"
  ),
  methods = list(
    initialize = function(num_trees = 10) {
      num_trees <<- num_trees
      trees <<- list()
      cat("Random Forest Classifier with", num_trees, "trees created.\n")
    },
    
    fit = function(X, y) {
      n <- nrow(X)
      m <- ncol(X)
      
      # Initialize feature importance matrix
      feature_importances_ <<- matrix(0, ncol = m, nrow = num_trees)
      
      for (i in 1:num_trees) {
        # Bootstrap sample
        idx <- sample(1:n, replace = TRUE)
        Xb <- X[idx, ]
        yb <- y[idx]
        
        # Build a tree (simple decision tree for demonstration)
        tree <- simple_decision_tree(Xb, yb, m)
        trees[[i]] <<- tree
        
        # Collect feature importance (simple counting of features used in splits)
        for (f in tree$features_used) {
          feature_importances_[i, f] <<- feature_importances_[i, f] + 1
        }
      }
    },
    
    predict = function(newdata) {
      predictions <- sapply(trees, function(tree) predict_tree(tree, newdata))
      apply(predictions, 2, function(p) names(which.max(table(p))))
    },
    
    getFeatureImportances = function() {
      rowMeans(feature_importances_)
    }
  )
)

# Helper function to build a simple decision tree (very basic)
simple_decision_tree <- function(X, y, m) {
  # Assume the tree is a stump, selecting one random feature and a random split
  feature_index <- sample(1:ncol(X), 1)
  split_value <- median(X[, feature_index])
  left <- y[X[, feature_index] <= split_value]
  right <- y[X[, feature_index] > split_value]
  return(list(
    split_feature = feature_index,
    split_value = split_value,
    left_class = ifelse(sum(left) > length(left)/2, 1, 0),
    right_class = ifelse(sum(right) > length(right)/2, 1, 0),
    features_used = feature_index
  ))
}

# Helper function to predict with a tree
predict_tree <- function(tree, newdata) {
  ifelse(newdata[, tree$split_feature] <= tree$split_value, tree$left_class, tree$right_class)
}



############################################
#### Random forest regressor
###########################################

RandomForestRegressor <- setRefClass(
  "RandomForestRegressor",
  # Simulate some data
  # set.seed(42)  # For reproducibility
  # X <- matrix(runif(100 * 4), ncol = 4)  
  # beta <- c(0.5, -1.2, 0.7, 2)        #
  # y <- X %*% beta + rnorm(100)          
  # Initialize the random forest regressor with a specified number of trees
  # regressor <- RandomForestRegressor$new(num_trees = 10)
  # Fit the model using the training data
  # regressor$fit(X, y)
  # Predict using the fitted model on the same data (typically you'd use new, unseen data)
  # predicted_values <- regressor$predict(X)
  # Print the predicted values
  # print(predicted_values)
  # Get feature importances
  # importances <- regressor$getFeatureImportances()
  # print(importances)
  fields = list(
    trees = "list",
    num_trees = "numeric",
    feature_importances_ = "matrix"
  ),
  methods = list(
    initialize = function(num_trees = 10) {
      num_trees <<- num_trees
      trees <<- list()
      cat("Random Forest Regressor with", num_trees, "trees created.\n")
    },
    
    fit = function(X, y) {
      n <- nrow(X)
      m <- ncol(X)
      
      # Initialize feature importance matrix
      feature_importances_ <<- matrix(0, ncol = m, nrow = num_trees)
      
      for (i in 1:num_trees) {
        
        idx <- sample(1:n, replace = TRUE)
        Xb <- X[idx, ]
        yb <- y[idx]
        

        tree <- simple_regression_tree(Xb, yb, m)
        trees[[i]] <<- tree
        
        # Collect feature importance (simple counting of features used in splits)
        for (f in tree$features_used) {
          feature_importances_[i, f] <<- feature_importances_[i, f] + 1
        }
      }
    },
    
    predict = function(newdata) {
      predictions <- sapply(trees, function(tree) predict_tree(tree, newdata))
      rowMeans(predictions)  # Average predictions from each tree
    },
    
    getFeatureImportances = function() {
      rowMeans(feature_importances_)
    }
  )
)

# Helper function to build a simple regression tree
simple_regression_tree <- function(X, y, m) {
  # Assume the tree is a stump, selecting one random feature and a random split
  feature_index <- sample(1:ncol(X), 1)
  split_value <- median(X[, feature_index])
  left <- y[X[, feature_index] <= split_value]
  right <- y[X[, feature_index] > split_value]
  return(list(
    split_feature = feature_index,
    split_value = split_value,
    left_value = mean(left),
    right_value = mean(right),
    features_used = feature_index
  ))
}

# Helper function to predict with a regression tree
predict_tree <- function(tree, newdata) {
  ifelse(newdata[, tree$split_feature] <= tree$split_value, tree$left_value, tree$right_value)
}


############################################
#### Adaboost classifier
###########################################

AdaBoostClassifier <- setRefClass(
  "AdaBoostClassifier",
  # Example usage of AdaBoostClassifier
  # Simulate some binary classification data
  #set.seed(123)
  # n <- 100
  # p <- 2
  # X <- matrix(runif(n * p), ncol = p)
  # y <- ifelse(X[, 1] + X[, 2] > 1, 1, -1)  # Simple linearly separable data
  # Create an instance of AdaBoostClassifier
  # classifier <- AdaBoostClassifier$new(num_rounds = 10)
  # Fit the model
  # classifier$fit(X, y)
  # Predict on the same dataset (usually should be on new data)
  # predictions <- classifier$predict(X)
  # Evaluate the predictions
  # table(Predicted = predictions, Actual = y)
  fields = list(
    classifiers = "list",
    alphas = "numeric",
    num_rounds = "numeric"
  ),
  methods = list(
    initialize = function(num_rounds = 10) {
      num_rounds <<- num_rounds
      classifiers <<- list()
      alphas <<- numeric(num_rounds)
      cat("AdaBoost Classifier with", num_rounds, "rounds created.\n")
    },
    
    fit = function(X, y) {
      if(any(is.na(X)) || any(is.na(y))) {
        stop("Data contains NA values. Please clean the data before fitting the model.")
      }
      n <- nrow(X)
      weights <- rep(1 / n, n)  # Initialize weights equally
      for (i in 1:num_rounds) {
        # Train a decision stump
        stump <- train_decision_stump(X, y, weights)
        # Predictions and error calculation
        predictions <- sapply(1:n, function(j) predict_stump(stump, X[j, ]))
        error <- sum(weights * (predictions != y))
        if (error > 0.5) break  # If error is greater than 50%, break the loop
        
        # Alpha calculation
        alpha <- 0.5 * log((1 - error) / error)
        alphas[i] <<- alpha
        classifiers[[i]] <<- stump
        
        # Update weights
        weights <- weights * exp(-alpha * y * predictions)
        weights <- weights / sum(weights)  # Normalize weights
      }
    },
    
    predict = function(newdata) {
      if (!is.matrix(newdata)) {
        newdata <- as.matrix(newdata)
      }
      if(any(is.na(newdata))) {
        stop("Prediction data contains NA values.")
      }
      
      final_predictions <- apply(newdata, 1, function(x) {
        sum(sapply(1:length(classifiers), function(i) {
          alphas[i] * predict_stump(classifiers[[i]], x)
        }))
      })
      sign(final_predictions)
    }
  )
)


train_decision_stump <- function(X, y, weights) {
  best_feature <- NULL
  best_threshold <- NULL
  best_inversion <- NULL
  min_error <- Inf
  

  for (f in 1:ncol(X)) {
    feature_values <- X[, f]
    thresholds <- sort(unique(feature_values))
    for (t in thresholds) {
      for (inversion in c(1, -1)) {
        predictions <- ifelse(feature_values * inversion < t * inversion, 1, -1)
        error <- sum(weights * (predictions != y))
        if (error < min_error) {
          best_feature <- f
          best_threshold <- t
          best_inversion <- inversion
          min_error <- error
        }
      }
    }
  }
  return(list(feature = best_feature, threshold = best_threshold, inversion = best_inversion))
}


predict_stump <- function(stump, x) {
  if (is.na(x[stump$feature])) {
    stop("Missing value detected in feature data during prediction.")
  }
  ifelse(x[stump$feature] * stump$inversion < stump$threshold * stump$inversion, 1, -1)
}

############################################
#### Adaboost Regressor
###########################################

train_regression_stump <- function(X, y, weights) {
  best_feature <- NULL
  best_threshold <- NULL
  best_split <- list(left_value = NULL, right_value = NULL)
  min_error <- Inf
  
  for (f in 1:ncol(X)) {
    feature_values <- X[, f]
    thresholds <- quantile(feature_values, probs = seq(0, 1, length.out = 10), na.rm = TRUE) # Handling potential NAs
    for (t in thresholds) {
      left_indices <- which(feature_values <= t)
      right_indices <- which(feature_values > t)
      if (length(left_indices) == 0 || length(right_indices) == 0) next # Skip this split if any side is empty
      left_value <- weighted_mean(y[left_indices], weights[left_indices])
      right_value <- weighted_mean(y[right_indices], weights[right_indices])
      predictions <- ifelse(feature_values <= t, left_value, right_value)
      error <- sum(weights * (predictions - y)^2)
      if (error < min_error && !is.na(error)) { # Check for NA in error
        min_error <- error
        best_feature <- f
        best_threshold <- t
        best_split$left_value <- left_value
        best_split$right_value <- right_value
      }
    }
  }
  list(feature = best_feature, threshold = best_threshold, split = best_split)
}

AdaBoostRegressor$methods(
  # Generate some data
  # set.seed(123)
  # X <- matrix(runif(100 * 2), ncol = 2)
  # y <- X[,1] * 2 + X[,2] * 3 + rnorm(100)
  # Create an AdaBoostRegressor instance
  # regressor <- AdaBoostRegressor$new(num_rounds = 10)
  # Fit the model
  # regressor$fit(X, y)
  # Predict on new data
  # new_X <- matrix(runif(10 * 2), ncol = 2)
  # predictions <- regressor$predict(new_X)
  # Show predictions
  # print(predictions)
  
  fit = function(X, y) {
    if(any(is.na(X)) || any(is.na(y))) {
      stop("Data contains NA values. Please clean the data before fitting the model.")
    }
    n <- nrow(X)
    weights <- rep(1 / n, n)  # Initialize weights equally
    for (i in 1:num_rounds) {
      # Train a simple regression stump
      stump <- train_regression_stump(X, y, weights)
      predictions <- predict_regression_stump(stump, X)
      errors <- abs(predictions - y)
      weighted_error <- sum(weights * errors) / sum(weights)  # Normalizing error
      
      # Avoid exact 0 or 1 errors
      weighted_error <- min(max(weighted_error, 1e-10), 1 - 1e-10)
      
      # Calculate alpha
      alpha <- 0.5 * log((1 - weighted_error) / weighted_error)
      alphas[i] <<- alpha
      regressors[[i]] <<- stump
      
      # Update weights, using alpha
      weights <- weights * exp(-alpha * (predictions - y)^2)
      weights <- weights / sum(weights)  # Normalize weights
    }
  }
)







