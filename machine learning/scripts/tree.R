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






